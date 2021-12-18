import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
from timeit import default_timer as timer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
ort_session = ort.InferenceSession("./model.onnx")

print("Loaded model")

num_layer = 9
num_attention_heads = 12
batch_size = 1
hidden_size = 768

def sample(session, input_text, num_tokens_to_produce = 30):    
    inputs = dict(tokenizer(input_text, return_tensors="np"))
    for step in range(num_tokens_to_produce):
        outputs = ort_session.run(None, inputs)
        next_token_logits = torch.from_numpy(outputs[0][:, -1, :])
        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        inputs['input_ids'] = np.array([np.append(inputs['input_ids'][0], next_tokens)])
        inputs['attention_mask'] = np.array([np.append(inputs['attention_mask'][0], [1])])
    return tokenizer.decode(inputs['input_ids'][0])

for i in range(10):
    start = timer()
    print(sample(ort_session, f"Here be dragons..."))
    end = timer()
    print(f"Test run {i + 1} took {end - start:.4f}s...")