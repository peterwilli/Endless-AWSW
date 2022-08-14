from jina import Executor, requests
from docarray import DocumentArray, Document
from .onnx_model_manager import OnnxModelManager
from .reply_processor import ReplyProcessor
import os
import json
import urllib.request
from typing import Dict
from profanity_check import predict as predict_profanity

filter_profanity = True
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, "model", "model.onnx")
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve("https://github.com/peterwilli/Endless-AWSW/releases/download/v0.3/model.onnx", model_path)
model_manager = OnnxModelManager(model_path)
reply_processor = ReplyProcessor()
command_retries = 5

class EndlessAWSWExec(Executor):
    """Replies as characters of the game Angels with Scaly Wings, a dragon-themed visual novel."""
    @requests
    def send_message(self, docs: DocumentArray, parameters: Dict, **kwargs):
        prompt = parameters['prompt']
        if filter_profanity and predict_profanity([prompt])[0] == 1:
            return DocumentArray(Document(text="profanity_detected", tags={'cmd': 'error'}))

        past_str = reply_processor.docs_to_string(docs)
        mods = 0
        if 'mods' in parameters:
            mods = parameters['mods']
        result = []
        for i in range(command_retries):
            reply = model_manager.say(past_str, prompt, do_sample = True, mods = mods)
            if filter_profanity and predict_profanity([reply])[0] == 1:
                continue

            if reply is not None:
                result = reply_processor.string_to_docs(model_manager.reply_prefix + reply)
                if len(result) > 0 and result[-1].tags['cmd'] != 'scn':
                    return DocumentArray(result)
                    
            return DocumentArray()

def test():
    m = EndlessAWSWExec()
    test_past = DocumentArray([
        Document(text = "Hey Remy!", tags = { 'cmd': 'msg', 'from': 'c' }),
        Document(text = "park2", tags = { 'cmd': 'scn' }),
        Document(text = "Hey!", tags = { 'cmd': 'msg', 'emotion': 'smile', 'from': 'Ry' }),
    ])
    print(f"json: {test_past.to_json()}")
    replies = m.send_message(test_past, {
        'prompt': 'Test' 
    })
    for reply in replies:
        print(f"Tags: {reply.tags} Reply: {reply.text}")
test()