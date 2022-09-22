from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
import numpy as np
import onnxruntime as ort
import random
import os
import math
import sys
from .validated_reply_buffer import ValidatedReplyBuffer, ValidationException

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
        return x / x.sum(axis=0, keepdims=1)
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(self.path))
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
        c = 1.0 - (scale * 0.5)
        for i in range(x.shape[0]):
            x[i] = c
            c *= scale
        return x
    
    def say_raw(self, prompt, do_sample = False, mods = 0) -> str:
        input_ids, attention_mask = self.get_model_input([prompt])
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        validated_reply_buffer = ValidatedReplyBuffer(
            mods = mods
        )
        for t in prompt:
            validated_reply_buffer.add_token(t, is_computer_generated = False)
            
        for step in range(self.max_length):
            inputs = {}
            inputs['attention_mask'] = attention_mask
            inputs['input_ids'] = input_ids
            outputs = self.model.run(None, inputs)
            sample_tries_left = 2
            while True:    
                amount_word_samples = 10
                next_token_logits = outputs[0][:, -1, :]
                next_tokens = np.argpartition(-next_token_logits, amount_word_samples).flatten()[:amount_word_samples]
                chances = next_token_logits.flatten()[next_tokens]
                chances = self.normalize(chances)
                chances_list = []
                for i, c in enumerate(chances):
                    chances_list.append({
                        'c': c,
                        'i': next_tokens[i]
                    })
                chances_list.sort(key=lambda x: x['c'], reverse=True)
                if validated_reply_buffer.in_message:
                    new_chances = np.zeros(10, dtype = np.float32)
                    self.word_chance(new_chances, 0.25)
                    for i in range(len(new_chances)):
                        new_chances[i] = new_chances[i] * chances_list[i]['c']
                    selection = random.choices(chances_list, weights=new_chances, k=1)[0]['i']
                    next_tokens = np.array([selection])
                else:
                    next_tokens = np.argmax(next_token_logits, axis=-1)
                
                if next_tokens[0] == eos_token_id:
                    # We end at eos_token as validated_reply_buffer doesn't track this token
                    return validated_reply_buffer.tokens

                token_str = self.tokenizer.decode(next_tokens)
                old_tokens = validated_reply_buffer.tokens
                try:
                    for t in token_str:
                        if validated_reply_buffer.add_token(t, is_computer_generated = True) == 1:
                            return validated_reply_buffer.tokens
                    break
                except ValidationException as e:
                    done_part = validated_reply_buffer.tokens[len(prompt):].strip()
                    logging.error(e)
                    if done_part.startswith("<") and done_part.endswith('"'):
                        logging.info(f"Validation exception, but we still have a valid reply ({done_part}), sending that instead...")
                        break
                    logging.warn(f"Validation exception with last tokens:\n{validated_reply_buffer.tokens}\nRetrying generate with last known working tokens:\n{old_tokens}")
                    validated_reply_buffer = ValidatedReplyBuffer(old_tokens)
                    sample_tries_left -= 1
                    if sample_tries_left == 0:
                        logging.warning("Can't find valid samples for message!")
                        return None
            # Update input_ids, attention_mask and past
            input_ids = np.array(self.tokenizer.encode(validated_reply_buffer.tokens), dtype=np.int64)
            input_ids = input_ids[np.newaxis, ...]
            attention_mask = np.ones((batch_size, input_ids.shape[1]), dtype=np.int64)
        return validated_reply_buffer.tokens
    
    def say(self, past, prompt, do_sample = False, mods = 0) -> str:
        prompt = f'{past}<p><msg>c "{prompt}"{self.reply_prefix}'
        raw_reply = self.say_raw(prompt, do_sample = do_sample, mods = mods)
        if raw_reply is None:
            return None
        return raw_reply[len(prompt):].strip()
    
if __name__ == "__main__":
    manager = OnnxModelManager("models/awsw_onnx/model_quant.onnx")
    test_prompt = "Hey sweetie!"
    reply = manager.say_raw(test_prompt)
    print(f"Test prompt: {test_prompt} Test reply: {reply}")