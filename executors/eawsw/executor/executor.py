from jina import Executor, requests
from docarray import DocumentArray, Document
from onnx_model_manager import OnnxModelManager
from reply_processor import ReplyProcessor
import os
import json
import urllib.request

model_path = os.path.join("model", "model.onnx")
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://github.com/peterwilli/Endless-AWSW/releases/download/v0.3/model.onnx", model_path)

model_manager = OnnxModelManager()
reply_processor = ReplyProcessor()
command_retries = 5

class EndlessAWSWExec(Executor):
    """Replies as characters of the game Angels with Scaly Wings, a dragon-themed visual novel."""
    @requests
    def send_message(self, docs: DocumentArray, **kwargs):
        doc = docs[0]
        past = json.loads(doc.text)
        past_str = reply_processor.commands_to_string(past)
        prompt = doc.tags['prompt']
        mods = 0
        if 'mods' in doc.tags:
            mods = doc.tags['mods']
        result = []
        for i in range(command_retries):
            reply = model_manager.say(past_str, prompt, do_sample = True, mods = mods)
            if reply is not None:
                result = reply_processor.string_to_commands(model_manager.reply_prefix + reply)
                if len(result) > 0 and result[-1]['cmd'] != 'scn':
                    return DocumentArray([Document(text=json.dumps(result))])
            return DocumentArray([Document(text=json.dumps([]))])