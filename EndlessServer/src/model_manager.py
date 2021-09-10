from transformers import Trainer, TrainingArguments
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import re
import sys
import logging

class ModelManager:
    def __init__(self, path):
        self.path = path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.allowed_tokens = {
            'Ry': 'Remy',
            'Lo': 'Lorem',
            'Br': 'Bryce',
            'Wr': 'Unknown name',
            'Ka': 'Katsuharu',
            'Rz': 'Reza',
            'Kv': 'Kevin',
            'Zh': 'Zhong',
            'm': 'Narrator',
            'An': 'Anna',
            'Ad': 'Adine',
            'Sb': 'Sebastian'
        }
        self.end_token = "<|endoftext|>"
        self.splitter = re.compile(r'\s|\"')

    def load_model(self):
        training_args = TrainingArguments(
            self.path,
            do_train = False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(self.path, pad_token_id = self.tokenizer.eos_token_id)
        model.to(self.device)
        model.resize_token_embeddings(len(self.tokenizer))
        model.eval()
        self.model = model

    def post_process_reply(self, reply):
        tokens = self.splitter.split(reply)
        from_character = None
        msg = []
        for token in tokens:
            if from_character is None and token in self.allowed_tokens:
                from_character = token
            elif token == self.end_token:
                break
            else:
                msg.append(token)
        if from_character is None:
            return None
        result = {
            'cmd': "msg",
            'from': from_character,
            'msg': " ".join(msg).strip()
        }
        return result

    def say(self, past, prompt):
        prompt = f'{past}{self.end_token}c \"{prompt}\"{self.end_token}DragonReply'
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt_tokens).unsqueeze(0)
        prompt_tensor = prompt_tensor.to(self.device)

        while True:
            sample_outputs = self.model.generate(
                prompt_tensor, 
                do_sample=True,   
                top_k=50, 
                max_length = 128,
                top_p=0.95, 
                num_return_sequences=1
            )

            text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=False)
            text_trimmed = text[len(prompt):]
            post_processed = self.post_process_reply(text_trimmed)
            logging.debug(f"prompt: {prompt} reply: {text} (trimmed: {text_trimmed}) ({len(prompt_tokens)} tokens) post_processed: {post_processed}")
            if post_processed is not None:
                return post_processed