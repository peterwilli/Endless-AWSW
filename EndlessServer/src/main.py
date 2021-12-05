from flask import Flask, json, request
from model_manager import ModelManager
from reply_processor import ReplyProcessor
import os
import json
import logging
logging.basicConfig(level=logging.DEBUG)

api = Flask(__name__)
model_manager = ModelManager(os.path.join("/", "opt", "awsw", "model"))
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
    reply = model_manager.say(past_str, prompt, top_k = 50, top_p = 0.7)
    possible_result = reply_processor.post_process_reply(model_manager.reply_prefix + reply)
    if possible_result is not None:
      result = possible_result
      break
  return {
    'cmds': result
  }

if __name__ == '__main__':
  api.run(host="0.0.0.0") 