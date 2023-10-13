from docarray import BaseDoc

class InputDoc(BaseDoc):
    past: str
    prompt: str

class OutputDoc(BaseDoc):
    reply: str
