from flask import Flask, json, request
from model_manager import ModelManager
from reply_processor import ReplyProcessor
import os
import logging
logging.basicConfig(level=logging.DEBUG)

api = Flask(__name__)
model_manager = ModelManager(os.path.join("/", "opt", "awsw", "model"))
reply_processor = ReplyProcessor()

@api.route('/get_command', methods=['GET'])
def get_command():
  past = request.args.get("past")
  prompt = request.args.get("prompt")
  reply = model_manager.say(past, prompt, top_k = 50, top_p = 0.7)
  result = reply_processor.post_process_reply(model_manager.reply_prefix + reply)
  return {
    'raw_reply': model_manager.reply_prefix + reply,
    'cmds': result
  }

if __name__ == '__main__':
  api.run(host="0.0.0.0") 