import numpy as np
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
# disallowed_sentences = [
#     "I want to have sex with you",
#     "I'll fuck you!",
#     "Show me your dick",
#     "Fuck the dragon",
#     "Show me your pussy",
#     "Can I see your penis/vagina",
#     "I want boobs"
# ]
# disallowed_embedded = np.stack(model.encode(disallowed_sentences), axis=0)

# def is_disallowed(sentence) -> bool:
#     embedded = model.encode([sentence])
#     return (embedded @ disallowed_embedded.T).max().item() > 0.7

# Temporary workaround (Error I had with this: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method)
def is_disallowed(sentence) -> bool:
    return False