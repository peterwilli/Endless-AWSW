from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
import numpy as np
import onnxruntime as ort
import torch

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
    
    def say_raw(self, prompt, top_k=None, top_p=None) -> str:
        input_ids, attention_mask, past = self.get_model_input([prompt])
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        all_token_ids = input_ids.copy()
        
        shape_name_mapping = {
            'sequence': input_ids.shape[1],
            'batch': batch_size,
            'past_sequence': 0,
            'past_sequence + sequence': input_ids.shape[1]
        }
        type_name_mapping = {
            'tensor(int64)': np.int64,
            'tensor(float)': np.float32
        }
        def map_shape(x):
            if type(x) is str:
                return shape_name_mapping[x]
            return x

        for step in range(self.max_length):
            inputs = {

            }
            for input in self.model.get_inputs():
                processed_shape = list(map(map_shape, input.shape))
                inputs[input.name] = np.zeros(processed_shape, dtype = type_name_mapping[input.type])

            for i in range(0, self.num_layer):
                inputs[f'past_key_values.{i}.key'] = np.ascontiguousarray(past[i * 2])
                inputs[f'past_key_values.{i}.value'] = np.ascontiguousarray(past[(i * 2) + 1])
            inputs['attention_mask'] = attention_mask
            inputs['input_ids'] = input_ids
            outputs = self.model.run(None, inputs)
            next_token_logits = outputs[0][:, -1, :]
            # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
            next_tokens = np.argmax(next_token_logits, axis=-1)
            all_token_ids = np.concatenate((all_token_ids, np.expand_dims(next_tokens, -1)), axis=-1)
            # Update input_ids, attention_mask, position_ids and past
            input_ids = next_tokens.reshape((batch_size, 1))   
            attention_mask = np.ones((batch_size, 1), dtype=np.float32)

            past = []
            for i in range(self.num_layer * 2):
                past_i = outputs[i + 1]
                past.append(past_i)
                
            if eos_token_id in next_tokens:
                break
        return self.tokenizer.decode(all_token_ids[0], skip_special_tokens=True)
    
    def say(self, past, prompt, top_k=None, top_p=None) -> str:
        prompt = f'{past}<p><msg>c "{prompt}"{self.reply_prefix}'
        return self.say_raw(prompt, top_k=top_k, top_p=top_p)[len(prompt):].strip()
    
if __name__ == "__main__":
    manager = OnnxModelManager("models/onnx/gpt-neo-default-past-lm.onnx")
    test_prompt = "Hey sweetie!"
    reply = manager.say_raw(test_prompt)
    print(f"Test prompt: {test_prompt} Test reply: {reply}")