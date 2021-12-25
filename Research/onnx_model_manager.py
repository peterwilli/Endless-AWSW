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
        
    def get_model_input(self, prompt):
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True)
        input_ids = np.array(encodings_dict['input_ids'])
        attention_mask = np.array(encodings_dict['attention_mask'], dtype=np.float32)

        #Empty Past State for generating first word
        empty_past = []
        batch_size = input_ids.shape[0]
        past_shape = [batch_size, 12, 0, 64]
        for i in range(self.num_layer * 2):
            empty_past.append(np.zeros(past_shape, dtype=np.float32))

        return input_ids, attention_mask, empty_past
    
    def word_chance(self, x, scale):
        c = 1.0 - (scale * 0.5)
        for i in range(x.shape[0]):
            x[i] = c
            c *= scale
        return x
    
    def say_raw(self, prompt, do_sample = False) -> str:
        input_ids, attention_mask, past = self.get_model_input([prompt])
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        all_token_ids = input_ids
        is_in_message = False
        for step in range(self.max_length):
            inputs = {}
            for i in range(0, self.num_layer):
                inputs[f'past_key_values.{i}.key'] = np.ascontiguousarray(past[i * 2])
                inputs[f'past_key_values.{i}.value'] = np.ascontiguousarray(past[(i * 2) + 1])
            inputs['attention_mask'] = attention_mask
            inputs['input_ids'] = input_ids
            outputs = self.model.run(None, inputs)                
            next_token_logits = outputs[0][:, -1, :]
            
            if do_sample:
                noise = np.random.uniform(low = 0.9, high = 1, size = next_token_logits.shape)
                next_token_logits = next_token_logits * noise
                next_tokens = np.argpartition(-next_token_logits, 20).flatten()[:20]
                chances = next_token_logits.flatten()[next_tokens]
                chances = self.normalize(chances)
                chances_list = []
                for i, c in enumerate(chances):
                    chances_list.append({
                        'c': c,
                        'i': next_tokens[i]
                    })
                chances_list.sort(key=lambda x: x['c'], reverse=True)
                dyn_chance = 0.0
                if is_in_message:
                    dyn_chance = 0.5
                new_chances = np.linspace(0, 1, 20)
                self.word_chance(new_chances, dyn_chance)
                if is_in_message:
                    for i in range(len(new_chances)):
                        new_chances[i] = new_chances[i] * chances_list[i]['c']
                selection = random.choices(chances_list, weights=new_chances, k=1)[0]['i']
                next_tokens = np.array([selection])
                #print([f"{self.tokenizer.decode(c['i'])} {new_chances[i]:.2f}" for i, c in enumerate(chances_list)], self.tokenizer.decode(next_tokens))
                if '"' in self.tokenizer.decode(next_tokens):
                    is_in_message = not is_in_message
                # print("weights: ", [f"{n:.2f}" for n in new_chances], self.tokenizer.decode(next_tokens), dyn_chance)
            else:
                next_tokens = np.argmax(next_token_logits, axis=-1)
            all_token_ids = np.concatenate((all_token_ids, np.expand_dims(next_tokens, -1)), axis=-1)
            # Update input_ids, attention_mask and past
            input_ids = next_tokens.reshape((batch_size, 1))   
            attention_mask = np.ones((batch_size, 1), dtype=np.float32)

            past = []
            for i in range(self.num_layer * 2):
                past_i = outputs[i + 1]
                past.append(past_i)
                
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