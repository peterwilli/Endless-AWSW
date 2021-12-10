import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
from timeit import default_timer as timer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
ort_session = ort.InferenceSession("models/onnx/eawsw/model.onnx")

print("Loaded model")

for i in range(10):
    start = timer()
    inputs = tokenizer(f"In my dreams I slowly transform in number {i}", return_tensors="np")
    outputs = ort_session.run(None, dict(inputs))
    end = timer()
    print(f"Test run {i + 1} took {end - start:.4f}s...")