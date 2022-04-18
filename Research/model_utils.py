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
import queue
import math
import threading
import re
from random import randrange
import multiprocessing
from scipy import interpolate
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import Trainer, TrainingArguments, TrainerCallback
from config import Config
import onnx
from onnx_model_manager import OnnxModelManager
from onnxruntime.quantization import quantize_dynamic, QuantType
    
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

class ModelSeeder:
    def __init__(self, tokenizer):
        self.main_model_path = os.path.join("models", "main_gptneo_onnx")
        if not os.path.exists(os.path.join(self.main_model_path, "model_quant.onnx")):
            os.system(f"python3 -m transformers.onnx --model='EleutherAI/gpt-neo-125M' --feature=causal-lm-with-past {self.main_model_path} --atol=0.0002")
            self.optimize_onnx()
        self.onnx_model_manager = OnnxModelManager(os.path.join(self.main_model_path, "model_quant.onnx"))
        self.tokenizer = tokenizer
        self.buffer = []
        self.done_queue = queue.Queue(maxsize = 10000)
        self.running = False
        self.buffer_thread = threading.Thread(target=self.buffer_worker)
        
    def start_worker(self):
        self.running = True
        self.buffer_thread.start()
        
    def stop_worker(self):
        self.running = False
        
    def buffer_worker(self):
        num_tokens = len(self.tokenizer)
        while self.running:
            random_input_id = random.randint(0, num_tokens)
            output = self.onnx_model_manager.say_raw(self.tokenizer.decode([random_input_id])[0], do_sample=True)
            self.done_queue.put(output)

    def optimize_onnx(self):
        model_fp32 = os.path.join(self.main_model_path, "model.onnx")
        model_quant = os.path.join(self.main_model_path, "model_quant.onnx")
        model_opt = os.path.join(self.main_model_path, "model-opt.onnx")
        quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
        os.system(f"rm {model_opt}")
        
    def seed_model(self, batch):
        input_ids = []
        attention_mask = []
        for i in range(len(batch['input_ids'])):
            input_ids.append(batch['input_ids'][i])
            attention_mask.append(batch['attention_mask'][i])
            # Add a text from the main model
            output = self.done_queue.get()
            output = self.tokenizer.encode(output)
            input_ids.append(output)
            attention_mask.append([1] * len(output))
        return {
            'attention_mask': attention_mask,
            'input_ids': input_ids
        }
        
def content_aware_encode(tokenizer, text) -> [int]:
    tokens = tokenizer.encode(text)
    new_tokens = []
    for token in tokens:
        if token == 6927: # ><
            new_tokens += [29, 27]
        else:
            new_tokens.append(token)
    return new_tokens
        
def get_dataset(tokenizer, path_train, block_size = 128):
    dataset = load_dataset('text', data_files={'train': path_train, 'test': os.path.join(Config.work_dir, "data_test.txt")})
    # model_seeder = ModelSeeder(tokenizer)
    # model_seeder.start_worker()
        
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
        
    def inject_random_rp(batch):
        last_scene = None
        last_character = None
        result = []
        
        re_msg = re.compile(r'([a-zA-Z]{1,2})\s"(.*?)"')
        for i, item in enumerate(batch['text']):
            if random.random() <= inject_rp_chance_pct:
                if item.startswith("<d>"):
                    msg_match = re_msg.search(item)
                    if msg_match is not None:
                        filtered_rp_list = []
                        msg_from = msg_match.group(1)
                        for rp_json in rp_list:
                            if 'about_character' in rp_json:
                                if msg_from != rp_json['about_character']:
                                    filtered_rp_list.append(rp_json)
                            else:
                                filtered_rp_list.append(rp_json)
                        if len(filtered_rp_list) > 0:
                            rp = random.choice(filtered_rp_list)
                            batch['text'][i] += rp['cmd'].strip()
        return batch
    
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
                item = item.replace("%lastcharactercode", last_character)
                item = item.replace("%lastcharacter", Config.interactable_characters[last_character])
            if not '%lastcharacter' in item and not '%lastscene' in item:
                result.append(item)
        return { 'text': result }

    def shuffle_groups(batch):
        # i = 0
        # result = []
        # while i < len(batch['text']):
        #     slice_size = random.randint(2, 10)
        #     result.append("".join(batch['text'][i:i + slice_size]))
        #     i = i + slice_size
        result = []
        last_starts_with_player = None
        tmp_list = []
        tmp_list2 = []
        for i in range(0, len(batch['text'])):
            line = batch['text'][i]
            starts_with_player = line.startswith("<p>")
            if last_starts_with_player is not None:
                if last_starts_with_player != starts_with_player:
                    tmp_list2.append(random.choice(tmp_list))
                    # We need to make a new batch
                    tmp_list = []
            tmp_list.append(line)                    
            last_starts_with_player = starts_with_player
        i = 0
        while i < len(tmp_list2):
            slice_size = random.randint(2, 10)
            result.append("".join(tmp_list2[i:i + slice_size]))
            i = i + slice_size
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
    #dataset_map_cores = 1
    dataset_batch_size = 1000

    class AWSWDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, dataset_type):
            self.current_dataset = dataset
            self.dataset_type = dataset_type
            self.current_idx = 0
            # Hack to avoid log spam. Map() doesn't have a way to turn off the logging
            # See: https://github.com/huggingface/datasets/issues/2651
            datasets.utils.set_progress_bar_enabled(False)
            self.shuffle()

        def shuffle(self):
            dataset = self.current_dataset.map(
                inject_random_rp,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
            dataset = dataset.map(
                shuffle_groups,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )
            dataset = dataset.shuffle(seed=random.randint(0, 999999))
            dataset = dataset.map(
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
                batch_size=dataset_batch_size,
                num_proc=dataset_map_cores
            )

        def __len__(self):
            return len(self.mapped_dataset[self.dataset_type])

        def __iter__(self):
            self.shuffle()
            return iter(self.mapped_dataset[self.dataset_type])
    
    return {
        # 'model_seeder': model_seeder,
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

    if not os.path.isfile(os.path.join(Config.work_dir, "data_train_sample.txt")):
        with open(os.path.join(Config.work_dir, "data_train_sample.txt"), "w") as f:
            for l in train_lines[:10]:
                f.write(l + "\n")
                
    if not os.path.isfile(os.path.join(Config.work_dir, "data_train.txt")):
        with open(os.path.join(Config.work_dir, "data_train.txt"), "w") as f:
            for l in train_lines:
                f.write(l + "\n")
                
            # flat_lines = split_branches(data).split("\n")
            # for l in flat_lines:
            #     f.write(l + "\n")

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
            self.no_grad_masks = self.make_no_grad_masks(0.01)
            self.named_parameters = list(model.named_parameters())
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

        def make_no_grad_masks(self, model_train_pct):
            masks = []
            for p in model.parameters():
                mask = torch.zeros(*p.shape)
                flattened_view = torch.flatten(mask)
                to_pick_len = math.floor(len(flattened_view) * model_train_pct)
                flattened_view[0:to_pick_len] = 1
                mask = mask.int().to(model.device)
                masks.append(mask)
            return masks
        
        # def on_before_optimizer_step(self, args, state, control, **kwargs):
        #     for i, w in enumerate(model.parameters()):    
        #         if w.grad is not None:
        #             w.grad *= self.no_grad_masks[i]
                    
    class AWSWTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            main_model, _ = get_model("EleutherAI/gpt-neo-125M")
            main_model.to(model.device)
            self.main_model = main_model
            self.params_len = len(list(model.parameters()))
            self.avg_loss_tries = 50
            self.last_avg_loss = None
            self.tick = 0
            self.mix_rate = 0.1
            self.loss_log = []
                
        def compute_loss(self, model, inputs, return_outputs=False):
            with torch.no_grad():
                for p1, p2 in zip(model.parameters(), self.main_model.parameters()):
                    diff = abs(p1.data - p2.data)
                    diff_mean = diff.mean()
                    learning_rate = optimizer.param_groups[0]['lr']
                    p1.data = torch.lerp(p1.data, p2.data, self.mix_rate)
                    
            outputs = model(**inputs)
            loss = outputs.get("loss")
            self.loss_log.append(loss.detach().cpu().numpy())
            avg_loss = 0
            if len(self.loss_log) == self.avg_loss_tries:
                avg_loss = sum(self.loss_log) / len(self.loss_log)
                if self.last_avg_loss is None:
                    self.last_avg_loss = avg_loss
                else:
                    #avg_loss_diff = abs(avg_loss - self.last_avg_loss)
                    #if avg_loss_diff > 0.0001:
                    if self.last_avg_loss < avg_loss:
                        # Loss gone up, time to stop mixing so much
                        self.mix_rate = max(0.0001, self.mix_rate * 0.5)
                    else:
                        # Loss gone down, we can keep mixing
                        self.mix_rate = min(0.5, self.mix_rate * 1.5)
                    self.last_avg_loss = avg_loss
                self.loss_log.pop(0)
            if not 'model_closeness_loss' in results:
                results['model_closeness_loss'] = []
            if not 'mix_rate' in results:
                results['mix_rate'] = []
            if not 'avg_loss' in results:
                results['avg_loss'] = []
            results['avg_loss'].append(avg_loss)
            results['mix_rate'].append(self.mix_rate)
            results['model_closeness_loss'].append(diff_mean.cpu().numpy())
            return (loss, outputs) if return_outputs else loss
                
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