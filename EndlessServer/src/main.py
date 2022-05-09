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
  logging.error(past)
  past_str = reply_processor.commands_to_string(past)
  prompt = request.args.get("prompt")
  result = []
  for i in range(command_retries):
    reply = model_manager.say(past_str, prompt, do_sample=True)
    if reply is not None:
      logging.debug(f"Reply before processing: {reply}")
      result = reply_processor.string_to_commands(model_manager.reply_prefix + reply)
      logging.debug(f"Reply after processing: {result}")
      if len(result) > 0 and result[-1]['cmd'] != 'scn':
        return {
          'cmds': result
        }
  return {
    'cmds': []
  }

if __name__ == '__main__':
  api.run(host="0.0.0.0") 