from transformers import Trainer, TrainingArguments
import torch
import re
import sys
import logging

class ReplyProcessor:
    def __init__(self):
        self.end_token = "<|endoftext|>"
        self.splitter = re.compile(r'\s')
        self.token_parser = re.compile(r'(<.*?>|[^<]*)')
        self.allowed_characters = {
            'c': 'Player',
            'Ry': 'Remy',
            'Lo': 'Lorem',
            'Ip': 'Ipsum',
            'Br': 'Bryce',
            'Wr': 'Unknown name',
            'Em': 'Emera',
            'Ka': 'Katsuharu',
            'Rz': 'Reza',
            'Kv': 'Kevin',
            'Zh': 'Zhong',
            'm': 'Narrator',
            'n': 'Back Story',
            'Mv': 'Maverick',
            'An': 'Anna',
            'Ad': 'Adine',
            'Sb': 'Sebastian'
        }
        self.allowed_commands = [
            "msg",
            "scn"
        ]

    def post_process_reply(self, reply):
        reply = f"{reply}"
        tokens = self.splitter.split(reply)