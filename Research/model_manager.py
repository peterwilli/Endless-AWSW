import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging

class ModelManager:
    def __init__(self, path = None, model = None, tokenizer = None, device = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.max_length = 128
        self.reply_prefix = "<d><scn>"
        if path is None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.path = path
            self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        model = AutoModelForCausalLM.from_pretrained(self.path)
        model.to(self.device)
        self.model = model
    
    def say_raw(self, prompt, top_k=None, top_p=None) -> str:
        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)

        sample_outputs = self.model.generate(
            generated, 
            do_sample=(top_k is not None and top_p is not None),
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            num_return_sequences=1
        )
        return self.tokenizer.decode(sample_outputs[0], skip_special_tokens=False)
    
    def say(self, past, prompt, top_k=None, top_p=None) -> str:
        prompt = f'{past}<p><msg>c "{prompt}"{self.reply_prefix}'
        return self.say_raw(prompt, top_k=top_k, top_p=top_p)[len(prompt):].strip()