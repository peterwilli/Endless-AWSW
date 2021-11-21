import torch
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPTNeoForCausalLM
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import Trainer, TrainingArguments, TrainerCallback
from config import Config
    
def get_model(name):
    tokenizer = GPT2Tokenizer.from_pretrained(name, eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = None
    if name == 'distilgpt2':
        model = GPT2LMHeadModel.from_pretrained(name, pad_token_id = tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    else:
        model = GPTNeoForCausalLM.from_pretrained(name, pad_token_id = tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    
    model.config.attention_dropout = 0.01
    model.config.embed_dropout = 0.01
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

    def map_dragon_reply_text(batch):
        result = {'text': []}
        for item in batch['text']:
            item_split = item.split(" ")
            player_replies = []
            dragon_replies = []
            current_reply = []
            handling_reply = None
            for token in item_split:
                if token == "PlayerReply":
                    if handling_reply is None:
                        handling_reply = "PlayerReply"
                    else:
                        if handling_reply == "PlayerReply":
                            # We need to store the PlayerReply
                            player_replies.append(" ".join(current_reply))
                            current_reply = []
                elif token == "DragonReply":
                    if handling_reply == "DragonReply":
                        # We need to store the DragonReply
                        dragon_replies.append(" ".join(current_reply))
                        current_reply = []

                    if handling_reply == "PlayerReply":
                        # We need to store the PlayerReply
                        player_replies.append(" ".join(current_reply))
                        current_reply = []

                    handling_reply = "DragonReply"
                    current_reply = []

                if handling_reply is not None:
                    current_reply.append(token)

            # There's always a dragon reply at the end.
            dragon_replies.append(" ".join(current_reply))
            for player_idx in range(len(player_replies)):
                for dragon_idx in range(len(dragon_replies)):
                    result['text'].append(player_replies[player_idx] + " " + dragon_replies[dragon_idx])

        return result

    dataset_map_cores = min(multiprocessing.cpu_count(), 10)
    dataset_batch_size = 1000

    dataset = dataset.map(
        map_dragon_reply_text,
        batched=True,
        batch_size=dataset_batch_size,
        num_proc=dataset_map_cores
    )

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=dataset_batch_size,
        remove_columns=["text"],
        num_proc=dataset_map_cores
    )

    return dataset.map(
        group_texts,
        batched=True,
        batch_size=dataset_batch_size,
        num_proc=dataset_map_cores
    )

def split_data(txt_file: str):
    with open(txt_file) as f:
        data = f.read()
    lines = data.split("\n")
    player_dragon_pairs = {}
    last_player_talk = []
    closed_player_talk = False
    re_player_talk = re.compile(r'c "(.*?)"')
    for line in lines:
        line = line.strip()
        line_split = line.split(" ")
        if len(line_split) <= 1:
            continue

        if line_split[0] == "c":
            if closed_player_talk:
                closed_player_talk = False
                last_player_talk = []
            last_player_talk.append(re.sub(re_player_talk, r"\1", line))
        else:
            if not closed_player_talk:
                last_player_talk = json.dumps(last_player_talk)
                if not last_player_talk in player_dragon_pairs:
                    player_dragon_pairs[last_player_talk] = []
                closed_player_talk = True

            line = "DragonReply " + line
            if last_player_talk is not None:
                player_dragon_pairs[last_player_talk].append(line)

    train_lines = []
    eval_lines = []
    eval_per_character = 0

    for player_line_str in player_dragon_pairs.keys():
        player_lines = json.loads(player_line_str)
        dragon_lines = player_dragon_pairs[player_line_str]
        compiled_line = " ".join([f'PlayerReply c "{player_line}"' for player_line in player_lines]) + " " + " ".join(dragon_lines)
        train_lines.append(compiled_line)

    test_bucket = {}
    for l in train_lines:
        l_split = l.split(" ")
        character = None
        for i, ls in enumerate(l_split):
            if ls == "DragonReply":
                character = l_split[i + 1]
                break
        if not character in test_bucket:
            test_bucket[character] = []
        test_bucket[character].append(l)

    for i in range(eval_per_character):
        for character in test_bucket.keys():
            random_line = test_bucket[character][randrange(len(test_bucket[character]))]
            eval_lines.append(random_line)
            for i2, t in enumerate(train_lines):
                if t == random_line:
                    del train_lines[i2]
                    break

    joined_eval_lines = "\n".join(eval_lines[:5])
    print(f"eval_lines: {joined_eval_lines}")
    joined_train_lines = "\n".join(train_lines[:5])
    print(f"train_lines: {joined_train_lines}")

    random.shuffle(train_lines)

    if not os.path.isfile(os.path.join(Config.work_dir, "data_train.txt")):
        with open(os.path.join(Config.work_dir, "data_train.txt"), "w") as f:
            for l in train_lines:
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
    lr = params['lr']
    batch_size = params['batch_size']
    train_len = len(dataset['train'])
    num_training_steps = math.ceil(train_len / batch_size)
    num_epoch = params['num_epoch']
    num_total_steps = num_training_steps * num_epoch
    num_warmup_steps = num_training_steps * params['warmup_factor']
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
                print(f"[{current_step}] set freeze_part_layers: {freeze_part_layers} (total layers: {len(named_parameters)})")
                to_freeze_count = params['to_freeze_count']
                for name, param in named_parameters[:to_freeze_count]:
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
            logging_steps=250,
            save_total_limit=2
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
        
    results['model'] = model
    results['tokenizer'] = tokenizer
    trainer_callback = AWSWTrainerCallback(optimizer, results)
    train(model, dataset, trainer_callback)
    del model
    del dataset
    del tokenizer
    del optimizer
    return None