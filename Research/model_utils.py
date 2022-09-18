import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
import pathlib
import json
import queue
import math
import threading
import re
from random import randrange
import multiprocessing
from scipy import interpolate
import datasets
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import Trainer, TrainingArguments, TrainerCallback
from config import Config
import onnx
from onnx_model_manager import OnnxModelManager
from onnxruntime.quantization import quantize_dynamic, QuantType
from reply_processor import ReplyProcessor
from regexes import *

reply_processor = ReplyProcessor()

def get_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tokenizer

def random_model_folder():
    now = int(time.time())
    models_dir = os.path.join(Config.work_dir, "models", str(now))
    if not os.path.isdir(models_dir):
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    return models_dir
        
def content_aware_encode(tokenizer, text) -> [int]:
    tokens = tokenizer.encode(text)
    new_tokens = []
    for token in tokens:
        if token == 6927: # ><
            new_tokens += [29, 27]
        else:
            new_tokens.append(token)
    return new_tokens
    
def get_dataset(seed, tokenizer, path_train, block_size = 128):
    dataset = load_dataset('text', data_files={'train': path_train, 'test': os.path.join(Config.work_dir, "data_test.txt")})
    
    def encode(batch):
        result = []
        attention_mask = []
        for item in batch['text']:
            tokens = content_aware_encode(tokenizer, item) + [tokenizer.eos_token_id]
            result.append(tokens)
            attention_mask.append([1] * len(tokens))
        return {
            'attention_mask': attention_mask,
            'input_ids': result
        }
    
    inject_rp_chance_pct = 0.5
    rp_list = None
    with open('rp_data.txt', 'r') as f:
        rp_list = [json.loads(line) for line in f.readlines()]
        
    inject_random_rp_random = random.Random(seed)
    def inject_random_rp(batch):
        result = []
        
        for i, item in enumerate(batch['text']):
            if inject_random_rp_random.random() <= inject_rp_chance_pct:
                cmds = reply_processor.string_to_commands(item)
                msg_from = None
                for cmd in cmds:
                    if cmd['cmd'] == 'msg':
                        if cmd['from'] in Config.interactable_characters:
                            msg_from = cmd['from']
                if msg_from is not None:
                    filtered_rp_list = []
                    for rp_json in rp_list:
                        if 'about_character' in rp_json:
                            if msg_from != rp_json['about_character']:
                                filtered_rp_list.append(rp_json)
                        else:
                            filtered_rp_list.append(rp_json)
                    
                    if len(filtered_rp_list) > 0:
                        rps_categorized = {}
                        for rp in filtered_rp_list:
                            if not rp['category'] in rps_categorized:
                                rps_categorized[rp['category']] = []
                            rps_categorized[rp['category']].append(rp)
                        category = inject_random_rp_random.choice(list(rps_categorized.keys()))
                        rp = inject_random_rp_random.choice(rps_categorized[category])
                        rp_cmd = rp['cmd'].strip()
                        if 'compatible_emotions' in rp:
                            for emotion in rp['compatible_emotions']:
                                if emotion in Config.valid_emotions_for_dragon[msg_from]:
                                    rp_cmd = rp_cmd.replace('normal "', f'{emotion} "')
                                    break
                        batch['text'][i] += rp_cmd
        return batch
    
    def gen_parse_variables():
        last_scene = None
        last_character = None
        
        def parse_variables(batch):
            nonlocal last_character, last_scene
            result = []

            for item in batch['text']:
                current_cmd = None
                for token in re_token.findall(item):
                    if len(token) > 0:
                        cmd_match = re_command.match(token)
                        if cmd_match is None:
                            if current_cmd == 'scn':
                                if not token.startswith("%"):
                                    last_scene = token
                            elif current_cmd == 'msg':
                                msg_match = re_msg.match(token)
                                if msg_match is None: 
                                    if not '%' in token:
                                        raise Exception(f"[parse_variables] Message not matched and doesn't contain variables! Item: {item} token: {token}")
                                else:
                                    msg_from = msg_match.group(1)
                                    if msg_from in Config.interactable_characters:
                                        last_character = msg_from
                        else:
                            current_cmd = cmd_match.group(1)

                if last_scene is not None:
                    item = item.replace("%lastscene", last_scene)
                if last_character is not None:
                    item = item.replace("%lastcharactercode", last_character)
                    item = item.replace("%lastcharacter", Config.interactable_characters[last_character])
                if not '%lastcharacter' in item and not '%lastscene' in item:
                    result.append(item)
                else:
                    raise Exception(f"Stray item: {item}")
            return { 'text': result }
        return parse_variables

    def shuffle_groups(batch):
        result = []
        last_character = None
        tmp_list = []
        tmp_list2 = []
        for i in range(0, len(batch['text'])):
            line = batch['text'][i].strip()
            if len(line) > 0:
                msg_match = re_msg.search(line)
                if msg_match is None:
                    raise Exception(f"msg_match None! Line: '{line}'")
                msg_from = msg_match.group(1)
                if last_character is not None:
                    if last_character != msg_from:
                        tmp_list_idxs = list(range(len(tmp_list)))
                        random.shuffle(tmp_list_idxs)
                        random_group_count = random.randint(1, len(tmp_list))
                        tmp_list_idxs = sorted(tmp_list_idxs[:random_group_count])
                        tmp_list2 += [tmp_list[idx] for idx in tmp_list_idxs]
                        # We need to make a new batch
                        tmp_list = [] 
                tmp_list.append(line)        
                last_character = msg_from
        i = 0
        while i < len(tmp_list2):
            slice_size = random.randint(2, 10)
            result.append("".join(tmp_list2[i:i + slice_size]))
            i = i + slice_size
        return { 'text': result }
    
    def filter_per_character(batch):
        result_texts = []
        for character in Config.interactable_characters:
            from_moments = []
            for idx, line in enumerate(batch['text']):
                line = line.strip()
                if len(line) == 0:
                    continue
                msg_match = re_msg.search(line)
                if msg_match is None:
                    raise Exception(f"msg_match None! Line: '{line}'")
                msg_from = msg_match.group(1)
                if msg_from == character:
                    from_moments.append(idx)
            moment = random.choice(from_moments)
            if moment > 0:
                text_before = batch['text'][moment - 1]
                result_texts.append(text_before)
            result_texts.append(batch['text'][moment])
        return { 'text': result_texts }

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Pad the end
        to_add = (math.ceil(total_length / block_size) * block_size) - total_length
        if to_add > 0:
            concatenated_examples['input_ids'] += [tokenizer.eos_token_id] * to_add
            concatenated_examples['attention_mask'] += [0] * to_add
            total_length += to_add
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset_map_cores = min(multiprocessing.cpu_count(), 1)
    # dataset_map_cores = 1
    dataset_batch_size = 1000
    # Make sure map is getting called over and over
    datasets.set_caching_enabled(False)

    class AWSWDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, dataset_type):
            self.current_dataset = dataset
            self.dataset_type = dataset_type
            self.random = np.random.RandomState(seed)
            datasets.logging.disable_progress_bar()
            self.shuffle()

        def shuffle(self):
            dataset = self.current_dataset.map(
                filter_per_character,
                batched=True,
                batch_size=10000000000000,
                num_proc=dataset_map_cores
            )
            dataset = dataset.map(
                shuffle_groups,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
            dataset = dataset.map(
                inject_random_rp,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
            dataset = dataset.map(
                gen_parse_variables(),
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
            dataset = dataset.shuffle(seed=self.random.randint(0, 2**32-1))
            dataset = dataset.map(
                encode,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores,
                remove_columns=["text"],
            )
            self.mapped_dataset = dataset.map(
                group_texts,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )

        def __len__(self):
            return len(self.mapped_dataset[self.dataset_type])

        def __iter__(self):
            self.shuffle()
            return iter(self.mapped_dataset[self.dataset_type])
    
    return {
        'train': AWSWDataset(dataset, 'train')
    }

def split_data(txt_file: str, shuffle_output = False):
    with open(txt_file) as f:
        data = f.read()
    lines = data.split("\n")
    train_lines = lines
    eval_lines = []
    
    if shuffle_output:
        random.shuffle(train_lines)
                
    if not os.path.isfile(os.path.join(Config.work_dir, "data_train.txt")):
        with open(os.path.join(Config.work_dir, "data_train.txt"), "w") as f:
            for l in train_lines:
                f.write(l + "\n")

    if not os.path.isfile(os.path.join(Config.work_dir, "data_test.txt")):
        with open(os.path.join(Config.work_dir, "data_test.txt"), "w") as f:
            for l in eval_lines:
                f.write(l + "\n")
    
def set_pretrained_model_dropout(h, dropout):
    for p in h:
        p.attn.attention.attn_dropout.p = dropout
        p.attn.attention.resid_dropout.p = dropout
        
def visualize_lr(params: dict):
    params = get_params(params)
    steps = 100
    optimizer = torch.optim.SGD([torch.tensor(1)], lr=1)
    scheduler = get_scheduler(optimizer, 5, steps, params)

    lrs = []
    for _ in range(steps):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.plot(lrs)
    plt.show()
    
def get_params(params: dict) -> dict:
    defaults = {
        "lr": 1e-4,
        "warmup_factor": 1,
        "scheduler": "polynomial_decay_schedule_with_warmup",
        "lr_end": 0.000002,
        "power": 0.6,
        "num_epoch": 10,
        "save_model": True,
        "batch_size": 32,
        "model_folder": os.path.join(Config.work_dir, "models", "awsw_main")
    }
    defaults.update(params)
    return defaults

def get_cycles_buildoff(
    optimizer, num_warmup_steps: int, num_training_steps: int, noise_amount: float = 0.0, num_cycles: int = 10, merge_cycles: int = 4, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    random_state = random.Random(94839)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        cycle_progress = float(num_cycles) * progress
        start_cycles = num_cycles - merge_cycles
        if cycle_progress > start_cycles:
            build_down_cycles = cycle_progress - start_cycles
            cycle_progress = start_cycles + (build_down_cycles / merge_cycles)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (cycle_progress % 1.0)))) + (random_state.uniform(-1, 1) * noise_amount)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(optimizer, num_warmup_steps: int, num_total_steps: int, params: dict):
    scheduler_str = params['scheduler']
    scheduler = None
    if scheduler_str == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps)
    elif scheduler_str == "cosine_with_hard_restarts_schedule_with_warmup":
        cycles = params['cycles']
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps, cycles)
    elif scheduler_str == "polynomial_decay_schedule_with_warmup":
        lr_end = params['lr_end']
        power = params['power']
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps, power=power, lr_end=lr_end)
    elif scheduler_str == "cycles_buildoff":
        cycles = params['cycles']
        scheduler = get_cycles_buildoff(optimizer, num_warmup_steps, num_total_steps, num_cycles = cycles, merge_cycles = 0, noise_amount = 0.01)
    return scheduler
        
def train_model(model, tokenizer, dataset, params: dict, results: dict):
    params = get_params(params)
    lr = params['lr']
    batch_size = params['batch_size']
    train_len = len(dataset['train'])
    num_steps_per_epoch = math.ceil(train_len / batch_size)
    num_epoch = params['num_epoch']
    num_total_steps = num_steps_per_epoch * num_epoch
    num_warmup_steps = num_steps_per_epoch * params['warmup_factor']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_total_steps, params)

    class AWSWTrainerCallback(TrainerCallback):
        def __init__(self, optimizer, results):
            self.random = np.random.RandomState(params['seed'])
            self.old_freeze_part_layers = None
            self.optimizer = optimizer
            self.results = results
            self.named_parameters = list(model.named_parameters())
            self.random.shuffle(self.named_parameters)
            self.did_freeze = False
            
        def on_train_end(self, args, state, control, **kwargs):
            learning_rate_history = [h['learning_rate'] for h in state.log_history if 'learning_rate' in h]
            loss_history = [h['loss'] for h in state.log_history if 'loss' in h]
            self.results['loss_history'] = loss_history
            self.results['learning_rate_history'] = learning_rate_history
            
        def on_step_begin(self, args, state, control, **kwargs):
            current_step = state.global_step
            # Freeze a part
            freeze_part_layers = False
            freeze_once = False
            learning_rate = self.optimizer.param_groups[0]['lr']
            if 'freeze_layer_rate' in params:
                freeze_layer_rate = params['freeze_layer_rate']
                freeze_part_layers = learning_rate > freeze_layer_rate
            if 'freeze_from_steps' in params:
                freeze_part_layers = current_step > params['freeze_from_steps']
            if 'freeze_once' in params:
                freeze_once = params['freeze_once']
            if self.old_freeze_part_layers is not freeze_part_layers:
                if freeze_once and self.did_freeze:
                    return
                if 'to_freeze_gpt_blocks' in params:
                    param_slice = self.named_parameters
                    for name, param in param_slice:
                        param.requires_grad = False
                    for name, param in model.transformer.h.named_parameters():
                        param.requires_grad = True
                    to_freeze_gpt_blocks = params['to_freeze_gpt_blocks']
                    param_slice = model.transformer.h[:to_freeze_gpt_blocks]
                    print(f"[{current_step}] set freeze_part_layers: {freeze_part_layers} (freezing {len(param_slice)} out of {len(model.transformer.h)} gpt blocks.)")
                    for name, param in param_slice.named_parameters():
                        param.requires_grad = not freeze_part_layers
                if 'to_freeze_count' in params:
                    to_freeze_count = params['to_freeze_count']
                    param_slice = self.named_parameters[:to_freeze_count]
                    print(f"[{current_step}] set freeze_part_layers: {freeze_part_layers} (freezing {len(param_slice)} out of {len(self.named_parameters)} layers.)")
                    for name, param in param_slice:
                        param.requires_grad = not freeze_part_layers
                self.old_freeze_part_layers = freeze_part_layers   
                if freeze_part_layers:
                    self.did_freeze = True     
                
    def train(model, dataset, trainer_callback):
        num_log_steps = 100
        logging_steps = math.floor(max(num_total_steps, num_log_steps) / num_log_steps)
        print(f"train -> logging_steps: {logging_steps}")
        
        training_args = TrainingArguments(
            params['model_folder'],
            seed=params['seed'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoch,
            logging_steps=num_log_steps,
            save_total_limit=2,
            log_level="error",
            save_strategy = "steps" if params['save_model'] else "no"
        )
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=dataset['train'],
            optimizers=(optimizer, scheduler),
            callbacks=[trainer_callback]
        )
        checkpoint_dirs = [os.path.join(params['model_folder'], d) for d in os.listdir(params['model_folder']) if os.path.isdir(os.path.join(params['model_folder'], d))]
        if len(checkpoint_dirs) > 0:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            trainer.train(latest_checkpoint)
        else:
            trainer.train()
        del training_args
        del trainer
        gc.collect()
        try:
            torch.distributed.destroy_process_group()
        except:
            pass
        torch.cuda.empty_cache()
    trainer_callback = AWSWTrainerCallback(optimizer, results)
    train(model, dataset, trainer_callback)
    del model
    del dataset
    del tokenizer
    del optimizer
    return None