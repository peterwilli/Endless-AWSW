from flask import Flask, json, request
from onnx_model_manager import OnnxModelManager
from reply_processor import ReplyProcessor
import os
import json
import logging

api = Flask(__name__)
if api.debug:
  logging.basicConfig(level=logging.DEBUG)
model_manager = OnnxModelManager(os.path.join("/", "opt", "awsw", "model", "model.onnx"))
reply_processor = ReplyProcessor()
command_retries = 5

@api.route('/get_command', methods=['GET'])
def get_command():
  past = request.args.get("past")
  past = json.loads(past)
  past_str = reply_processor.commands_to_string(past)
  prompt = request.args.get("prompt")
  result = []
  for i in range(command_retries):
    reply = model_manager.say(past_str, prompt, do_sample=True)
    logging.debug(f"Reply before processing: {reply}")
    possible_result = reply_processor.post_process_reply(model_manager.reply_prefix + reply)
    if possible_result is not None:
      result = possible_result
      break
  return {
    'cmds': result
  }

if __name__ == '__main__':
  api.run(host="0.0.0.0") 