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
        self.reply_prefix = "<d><scn>"
        if path is None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.path = path
            self.load_model()
            
    def content_aware_encode(self, text) -> [int]:
        tokens = self.tokenizer.encode(text)
        new_tokens = []
        for token in tokens:
            if token == 6927: # ><
                new_tokens += [29, 27]
            else:
                new_tokens.append(token)
        return new_tokens
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        model = AutoModelForCausalLM.from_pretrained(self.path)
        model.to(self.device)
        self.model = model
    
    @torch.cuda.amp.autocast()
    def say_raw(self, prompt, top_k=None, top_p=None, max_length:int=128) -> str:
        generated = torch.tensor(self.content_aware_encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)
        generated = {'input_ids': generated, 'attention_mask': torch.zeros_like(generated).fill_(1)}
        sample_outputs = self.model.generate(
            **generated, 
            do_sample=(top_k is not None and top_p is not None),
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_length,
            num_return_sequences=1
        )
        return self.tokenizer.decode(sample_outputs[0], skip_special_tokens=False)
    
    def say(self, past, prompt, top_k=None, top_p=None, max_length:int=128) -> str:
        prompt = f'{past}<p><msg>c "{prompt}"{self.reply_prefix}'
        return self.say_raw(prompt, top_k=top_k, top_p=top_p, max_length=max_length)[len(prompt):].strip()