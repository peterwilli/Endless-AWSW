from flask import Flask, json, request
from model_manager import ModelManager
import os
import logging
logging.basicConfig(level=logging.DEBUG)

api = Flask(__name__)
model_manager = ModelManager(os.path.join("/", "opt", "awsw", "model"))

@api.route('/get_command', methods=['GET'])
def get_command():
  past = request.args.get("past")
  prompt = request.args.get("prompt")
  result = model_manager.say(past, prompt)
  return result

if __name__ == '__main__':
    
    api.run(host="0.0.0.0") 