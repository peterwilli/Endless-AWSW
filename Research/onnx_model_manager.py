from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
import numpy as np
import onnxruntime as ort
import torch
import random
import math

class OnnxModelManager:
    def __init__(self, path = None, model = None, tokenizer = None, device = None):
        self.max_length = 128
        self.reply_prefix = "<d><scn>"
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layer = 12
        if path is None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.path = path
            self.load_model()
            
    def normalize(self, x):
        x = abs(np.min(x)) + x
        return x / x.sum(axis=0,keepdims=1)
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = ort.InferenceSession(self.path)
        
    def batch_content_aware_encode(self, texts) -> dict:
        encodings_dict = self.tokenizer.batch_encode_plus(texts, padding=True)
        new_batch = {
            'input_ids': [],
            'attention_mask': []
        }
        for i, tokens in enumerate(encodings_dict['input_ids']):
            new_tokens = []
            for token in tokens:
                if token == 6927: # ><
                    new_tokens += [29, 27]
                else:
                    new_tokens.append(token)
            new_batch['input_ids'].append(new_tokens)
            new_batch['attention_mask'].append([1] * len(new_tokens))
        return new_batch
        
    def get_model_input(self, prompt):
        encodings_dict = self.batch_content_aware_encode(prompt)
        input_ids = np.array(encodings_dict['input_ids'], dtype=np.int64)
        attention_mask = np.array(encodings_dict['attention_mask'], dtype=np.int64)
        return input_ids, attention_mask
    
    def word_chance(self, x, scale):
        c = 1.0
        for i in range(x.shape[0]):
            x[i] = c
            c *= scale
        return x
    
    def say_raw(self, prompt, do_sample = False, reply_as = None) -> str:
        input_ids, attention_mask = self.get_model_input([prompt])
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        all_token_ids = input_ids
        is_in_message = False
        for step in range(self.max_length):
            inputs = {}
            inputs['attention_mask'] = attention_mask
            inputs['input_ids'] = input_ids
            outputs = self.model.run(None, inputs)                
            next_token_logits = outputs[0][:, -1, :]
            
            if do_sample:
                next_tokens = np.argpartition(-next_token_logits, 10).flatten()[:10]
                if is_in_message:
                    chances = next_token_logits.flatten()[next_tokens]
                    chances = self.normalize(chances)
                    chances_list = []
                    for i, c in enumerate(chances):
                        chances_list.append({
                            'c': c,
                            'i': next_tokens[i]
                        })
                    chances_list.sort(key=lambda x: x['c'], reverse=True)
                    new_chances = np.zeros(10, dtype = np.float32)
                    self.word_chance(new_chances, 0.45)
                    for i in range(new_chances.shape[0]):
                        new_chances[i] = new_chances[i] * chances_list[i]['c']
                    selection = random.choices(chances_list, weights=new_chances, k=1)[0]['i']
                    next_tokens = np.array([selection])
                else:
                    next_tokens = np.argmax(next_token_logits, axis=-1)
                if '"' in self.tokenizer.decode(next_tokens):
                    is_in_message = not is_in_message
            else:
                next_tokens = np.argmax(next_token_logits, axis=-1)
            all_token_ids = np.concatenate((all_token_ids, np.expand_dims(next_tokens, -1)), axis=-1)
            # Update input_ids, attention_mask and past
            input_ids = all_token_ids
            attention_mask = np.ones((batch_size, 1), dtype=np.int64)
                
            if eos_token_id in next_tokens:
                break
        return self.tokenizer.decode(all_token_ids[0], skip_special_tokens=False)
    
    def say(self, past, prompt, do_sample=False) -> str:
        prompt = f'{past}<p><msg>c "{prompt}"{self.reply_prefix}'
        return self.say_raw(prompt, do_sample = do_sample)[len(prompt):].strip()
    
if __name__ == "__main__":
    manager = OnnxModelManager("models/awsw_onnx/model_quant.onnx")
    test_prompt = "Hey sweetie!"
    reply = manager.say_raw(test_prompt)
    print(f"Test prompt: {test_prompt} Test reply: {reply}")