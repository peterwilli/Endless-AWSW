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
from transformers import Trainer, TrainingArguments, TrainerCallback, ProgressCallback
from config import Config
import onnx
from onnx_model_manager import OnnxModelManager
from onnxruntime.quantization import quantize_dynamic, QuantType
from reply_processor import ReplyProcessor
from regexes import *

reply_processor = ReplyProcessor()

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
            slice_size = random.randint(4, 16)
            result.append("".join(tmp_list2[i:i + slice_size]))
            i = i + slice_size
        return { 'text': result }
    
    def filter_per_character(batch):
        result = []
        from_moments = {}
        for character in Config.interactable_characters:
            from_moments[character] = []
            for idx, line in enumerate(batch['text']):
                line = line.strip()
                if len(line) == 0:
                    continue
                msg_match = re_msg.search(line)
                if msg_match is None:
                    raise Exception(f"msg_match None! Line: '{line}'")
                msg_from = msg_match.group(1)
                if msg_from == character:
                    from_moments[msg_from].append(idx) 

        while True:
            for character in Config.interactable_characters:
                if len(from_moments[character]) > 0:
                    text_to_add = ""
                    moment = random.choice(from_moments[character])
                    if moment >= 4:
                        before_slice = random.randint(4, min(moment, 8))
                        start = moment - before_slice
                        text_before = "".join(batch['text'][start:moment])
                        if len(text_before) > 0:
                            text_to_add += text_before
                    text_to_add += batch['text'][moment]
                    from_moments[character].remove(moment)
                    result.append(text_to_add)
                    if len(from_moments[character]) == 0:
                        return { 'text': result }

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

    class AWSWDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, dataset_type):
            self.current_dataset = dataset
            self.dataset_type = dataset_type
            self.trim_len = 2000
            self.random = np.random.RandomState(seed)
            datasets.logging.disable_progress_bar()
            
        def create_shuffled_dataset(self):
            dataset = self.current_dataset.map(
                shuffle_groups,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
#             dataset = dataset.map(
#                 filter_per_character,
#                 batched=True,
#                 batch_size=9999999,
#                 num_proc=dataset_map_cores
#             )
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
            dataset = dataset.map(
                encode,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores,
                remove_columns=["text"],
            )
            dataset = dataset.shuffle(seed=random.randint(0, 2**32-1))
            return dataset.map(
                group_texts,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )

        def __len__(self):
            return self.trim_len
        
        def __iter__(self):
            return iter(list(self.create_shuffled_dataset()[self.dataset_type])[:self.trim_len])
                
    # Make sure map is getting called over and over
    datasets.disable_caching()
    dataset_orig = load_dataset('text', data_files={'train': path_train, 'test': os.path.join(Config.work_dir, "data_test.txt")})
    return {
        'train': AWSWDataset(dataset_orig, 'train')
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
    steps = 1000
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
        "gradient_accumulation_steps": 4,
        "batch_size": 32,
        "model_folder": os.path.join(Config.work_dir, "models", "awsw_main")
    }
    defaults.update(params)
    return defaults

def get_cycles_buildoff(
    optimizer, num_warmup_steps: int, num_training_steps: int, noise_amount: float = 0.0, num_cycles: int = 10, merge_cycles: int = 4, last_epoch: int = -1
):
    random_state = random.Random(94839)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        cycle_progress = float(num_cycles) * progress
        start_cycles = num_cycles - merge_cycles
        step_noise = noise_amount
        if cycle_progress > start_cycles:
            build_down_cycles = cycle_progress - start_cycles
            cycle_progress = start_cycles + (build_down_cycles / merge_cycles)
            # During buildoff there will be no noise
            step_noise = 0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (cycle_progress % 1.0)))) + (random_state.uniform(-1, 1) * step_noise)

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
        merge_cycles = params['merge_cycles']
        scheduler = get_cycles_buildoff(optimizer, num_warmup_steps, num_total_steps, num_cycles = cycles, merge_cycles = merge_cycles, noise_amount = 0.01)
    return scheduler
        
def train_model(model, tokenizer, dataset, params: dict, results: dict, callbacks = []):
    params = get_params(params)
    lr = params['lr']
    gradient_accumulation_steps = params['gradient_accumulation_steps']
    batch_size = params['batch_size']
    num_epoch = params['num_epoch']
    num_total_steps = math.ceil((len(dataset['train']) * num_epoch) / batch_size / gradient_accumulation_steps)
    num_warmup_steps = math.ceil(num_total_steps * params['warmup_factor'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"num_total_steps: {num_total_steps} num_warmup_steps: {num_warmup_steps}")
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_total_steps, params)

    class AWSWTrainerCallback(TrainerCallback):
        def __init__(self, results):
            self.results = results
            
        def on_train_end(self, args, state, control, **kwargs):
            learning_rate_history = [h['learning_rate'] for h in state.log_history if 'learning_rate' in h]
            loss_history = [h['loss'] for h in state.log_history if 'loss' in h]
            self.results['loss_history'] = loss_history
            self.results['learning_rate_history'] = learning_rate_history
            
    def train(model, dataset, trainer_callback):
        training_args = TrainingArguments(
            params['model_folder'],
            seed=params['seed'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epoch,
            logging_steps=1,
            fp16=True,
            save_total_limit=2,
            save_strategy = "steps" if params['save_model'] else "no",
        )
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=dataset['train'],
            optimizers=(optimizer, scheduler),
            callbacks=callbacks + [trainer_callback]
        )
        # To avoid log spam
        trainer.callback_handler.callbacks.pop()
        checkpoint_dirs = [os.path.join(params['model_folder'], d) for d in os.listdir(params['model_folder']) if os.path.isdir(os.path.join(params['model_folder'], d))]
        # silence the warnings. Please re-enable for inference!
        model.config.use_cache = False
        if len(checkpoint_dirs) > 0:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            trainer.train(latest_checkpoint)
        else:
            trainer.train()
        model.config.use_cache = True
        del training_args
        del trainer
        gc.collect()
        try:
            torch.distributed.destroy_process_group()
        except:
            pass
        torch.cuda.empty_cache()
    trainer_callback = AWSWTrainerCallback(results)
    train(model, dataset, trainer_callback)
    del model
    del dataset
    del tokenizer
    del optimizer
    return None