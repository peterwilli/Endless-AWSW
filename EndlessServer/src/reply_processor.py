from transformers import Trainer, TrainingArguments
import torch
import re
import sys
import logging

class ReplyProcessor:
    def __init__(self):
        self.end_token = "<|endoftext|>"
        self.splitter = re.compile(r'\s|\"')
        self.allowed_tokens = {
            'Ry': 'Remy',
            'Lo': 'Lorem',
            'Br': 'Bryce',
            'Wr': 'Unknown name',
            'Ka': 'Katsuharu',
            'Rz': 'Reza',
            'Kv': 'Kevin',
            'Zh': 'Zhong',
            'm': 'Narrator',
            'An': 'Anna',
            'Ad': 'Adine',
            'Sb': 'Sebastian',
            'Mv': 'Maverick'
        }

    def post_process_reply(self, reply):
        tokens = self.splitter.split(reply)
        from_character = None
        msg = []
        for token in tokens:
            if from_character is None and token in self.allowed_tokens:
                from_character = token
            elif token == self.end_token:
                break
            else:
                msg.append(token)
        if from_character is None:
            return None
        result = {
            'cmd': "msg",
            'from': from_character,
            'msg': " ".join(msg).strip()
        }
        return result