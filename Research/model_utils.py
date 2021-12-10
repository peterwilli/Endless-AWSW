import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
import seaborn as sns
import pandas as pd
import os
import gc
import pathlib
import json
import math
import re
from random import randrange
import multiprocessing
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import Trainer, TrainingArguments, TrainerCallback
from config import Config
    
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

def get_dataset(tokenizer, block_size = 128):
    dataset = load_dataset('text', data_files={'train': os.path.join(Config.work_dir, "data_train.txt"), 'test': os.path.join(Config.work_dir, "data_test.txt")})

    def encode(batch):
        result = []
        attention_mask = []
        for item in batch['text']:
            tokens = tokenizer.encode(item) + [tokenizer.eos_token_id]
            result.append(tokens)
            attention_mask.append([1] * len(tokens))
        return {
            'attention_mask': attention_mask,
            'input_ids': result
        }
    
    def parse_variables(batch):
        last_scene = None
        last_character = None
        result = []
        
        re_token = re.compile(r'(<.*?>|[^<]*)')
        re_command = re.compile(r'^<(.*?)>$')
        re_msg = re.compile(r'([a-zA-Z]{1,2})\s"(.*?)"')
        
        for item in batch['text']:
            current_cmd = None
            for token in re_token.findall(item):
                cmd_match = re_command.match(token)
                if cmd_match is None:
                    if current_cmd == 'scn':
                        if not token.startswith("%"):
                            last_scene = token
                    elif current_cmd == 'msg':
                        msg_match = re_msg.match(token)
                        if msg_match is not None:
                            msg_from = msg_match.group(1)
                            if msg_from in Config.interactable_characters:
                                last_character = msg_from
                else:
                    current_cmd = cmd_match.group(1)
                        
            if last_scene is not None:
                item = item.replace("%lastscene", last_scene)
            if last_character is not None:
                item = item.replace("%lastcharacter", Config.interactable_characters[last_character])
            if not '%lastcharacter' in item and not '%lastscene' in item:
                result.append(item)
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

    dataset_map_cores = min(multiprocessing.cpu_count(), 10)
    dataset_batch_size = 1000

    class AWSWDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, dataset_type):
            self.current_dataset = dataset
            self.dataset_type = dataset_type
            self.current_idx = 0
            self.shuffle()

        def shuffle(self):
            self.current_dataset = self.current_dataset.shuffle()
            # Hack to avoid log spam. Map() doesn't have a way to turn off the logging
            # See: https://github.com/huggingface/datasets/issues/2651
            datasets.utils.set_progress_bar_enabled(False)
            dataset = self.current_dataset.map(
                parse_variables,
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
            self.mapped_dataset = dataset.map(
                group_texts,
                batched=True,
                batch_size=100,
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

def split_branches(data):
    result = []
    quote_counter = 0
    line = ""
    for i in range(len(data)):
        if data[i] == "\n":
            continue
        line += data[i]
        if data[i] == '"':
            quote_counter += 1
        if quote_counter == 2:
            quote_counter = 0
            result.append(line.strip())
            line = ""
    return "\n".join(result)

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
                
            flat_lines = split_branches(data).split("\n")
            for l in flat_lines:
                f.write(l + "\n")

    if not os.path.isfile(os.path.join(Config.work_dir, "data_test.txt")):
        with open(os.path.join(Config.work_dir, "data_test.txt"), "w") as f:
            for l in eval_lines:
                f.write(l + "\n")
    
def set_pretrained_model_dropout(h, dropout):
    for p in h:
        p.attn.attention.attn_dropout.p = dropout
        p.attn.attention.resid_dropout.p = dropout
        
def train_model(model, tokenizer, dataset, params: dict, results: dict):
    defaults = {
        "lr": 1e-4,
        "warmup_factor": 1,
        "scheduler": "polynomial_decay_schedule_with_warmup",
        "lr_end": 0.000002,
        "power": 0.6,
        "freeze_layer_rate": 0.0009,
        "num_epoch": 10,
        "save_model": True,
        "batch_size": 32,
        "model_folder": os.path.join(Config.work_dir, "models", "awsw_main")
    }
    defaults.update(params)
    params = defaults
    lr = params['lr']
    batch_size = params['batch_size']
    train_len = len(dataset['train'])
    num_steps_per_epoch = math.ceil(train_len / batch_size)
    num_epoch = params['num_epoch']
    num_total_steps = num_steps_per_epoch * num_epoch
    num_warmup_steps = num_steps_per_epoch * params['warmup_factor']
    optimizer = AdamW(model.parameters(), lr=lr)
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

    class AWSWTrainerCallback(TrainerCallback):
        def __init__(self, optimizer, results):
            self.random = np.random.RandomState(params['seed'])
            self.old_freeze_part_layers = None
            self.optimizer = optimizer
            self.results = results
            self.named_parameters = list(model.named_parameters())
            self.random.shuffle(self.named_parameters)

        def on_train_end(self, args, state, control, **kwargs):
            learning_rate_history = [h['learning_rate'] for h in state.log_history if 'learning_rate' in h]
            loss_history = [h['loss'] for h in state.log_history if 'loss' in h]
            self.results['loss_history'] = loss_history
            self.results['learning_rate_history'] = learning_rate_history

        def on_step_begin(self, args, state, control, **kwargs):
            current_step = state.global_step
            # Freeze a part
            learning_rate = self.optimizer.param_groups[0]['lr']
            freeze_layer_rate = params['freeze_layer_rate']
            freeze_part_layers = learning_rate > freeze_layer_rate
            if 'freeze_from_steps' in params:
                freeze_part_layers = current_step > params['freeze_from_steps']
            if self.old_freeze_part_layers is not freeze_part_layers:
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
                
    def train(model, dataset, trainer_callback):
        training_args = TrainingArguments(
            params['model_folder'],
            seed=params['seed'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoch,
            logging_steps=math.floor(max(num_total_steps, 100) / min(num_total_steps, 100)),
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
        trainer.save_model()
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