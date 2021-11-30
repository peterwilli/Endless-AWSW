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

class Model(nn.Module):
    def __init__(self, base_model):
        super(Model, self).__init__()
        self.base_model = base_model
        self.extension = nn.Sequential(
            nn.Linear(100, 100)
        )

    def forward(self, x):
        return self.base_model(x)
    
def get_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name, eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = AutoModelForCausalLM.from_pretrained(name, pad_token_id = tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    model.config.attention_dropout = 0.1
    model.config.embed_dropout = 0.1
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def random_model_folder():
    now = int(time.time())
    models_dir = os.path.join(Config.work_dir, "models", str(now))
    if not os.path.isdir(models_dir):
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    return models_dir

def get_dataset(tokenizer, block_size):
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

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Pad the end
        to_add = (math.ceil(total_length / block_size) * block_size) - total_length
        if to_add > 0:
            concatenated_examples['input_ids'] += [tokenizer.pad_token_id] * to_add
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
            datasets.utils.disable_progress_bar()
            self.mapped_dataset = self.current_dataset.map(
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

#     dataset = dataset.map(
#         map_dragon_reply_text,
#         batched=True,
#         batch_size=dataset_batch_size,
#         num_proc=dataset_map_cores
#     )

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=dataset_batch_size,
        num_proc=dataset_map_cores,
        remove_columns=["text"],
    )

#     dataset = dataset.map(
#         group_texts,
#         batched=True,
#         batch_size=dataset_batch_size,
#         num_proc=dataset_map_cores
#     )
    
    return {
        'train': AWSWDataset(dataset, 'train')
    }

def split_branches(data):
    result = ""
    quote_counter = 0
    data = data.strip()
    for i in range(len(data)):
        if data[i] == "\n":
            continue
        result += data[i]
        if data[i] == '"':
            quote_counter += 1
        if quote_counter == 2:
            quote_counter = 0
            result += "\n"
    return result

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
    
                
def train_model(params: dict, results: dict, device):
    defaults = {
        "model_name": "distilgpt2",
        "lr": 1e-4,
        "warmup_factor": 1,
        "scheduler": "polynomial_decay_schedule_with_warmup",
        "lr_end": 0.000002,
        "power": 0.6,
        "freeze_layer_rate": 0.0009,
        "num_epoch": 10,
        "block_size": 128,
        "save_model": True,
        "batch_size": 32,
        "model_folder": os.path.join(Config.work_dir, "models", "awsw_main")
    }
    defaults.update(params)
    params = defaults
    model_name = params['model_name']
    model, tokenizer = get_model(model_name)
    model = model.to(device)
    named_parameters = list(model.named_parameters())
    dataset = get_dataset(tokenizer, params['block_size'])
    print("Dataset demo snapshot:")
    demo_idx = 0
    for item in dataset['train']:
        print(tokenizer.decode(item['input_ids']))
        if demo_idx > 0:
            break
        demo_idx += 1
    del demo_idx
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
            self.old_freeze_part_layers = None
            self.optimizer = optimizer
            self.results = results

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
                to_freeze_count = params['to_freeze_count']
                param_slice = named_parameters[:to_freeze_count]
                print(f"[{current_step}] set freeze_part_layers: {freeze_part_layers} (freezing {len(param_slice)} out of {len(named_parameters)} layers.)")
                for name, param in param_slice:
                    param.requires_grad = not freeze_part_layers
                self.old_freeze_part_layers = freeze_part_layers

    def train(model, dataset, trainer_callback):
        model.train()
        training_args = TrainingArguments(
            params['model_folder'],
            seed=params['seed'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoch,
            logging_steps=max(1, math.floor(num_total_steps / 100)),
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
        
    results['model'] = model
    results['tokenizer'] = tokenizer
    trainer_callback = AWSWTrainerCallback(optimizer, results)
    train(model, dataset, trainer_callback)
    del model
    del dataset
    del tokenizer
    del optimizer
    return None