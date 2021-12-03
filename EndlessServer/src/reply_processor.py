from transformers import Trainer, TrainingArguments
import torch
import re
import sys
import logging

class ReplyProcessor:
    def __init__(self):
        self.end_token = "<|endoftext|>"
        self.splitter = re.compile(r'\s')
        self.re_token = re.compile(r'(<.*?>|[^<]*)')
        self.re_command = re.compile(r'^<(.*?)>$')
        self.re_msg = re.compile(r'([a-zA-Z]{1,2})\s"(.*?)"')
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
        result = []
        current_cmd = None
        for token in self.re_token.findall(reply):
            cmd_match = self.re_command.match(token)
            if cmd_match is None:
                if current_cmd['cmd'] == 'scn':
                    current_cmd['scn'] = token
                    result.append(current_cmd)
                elif current_cmd['cmd'] == 'msg':
                    msg_match = self.re_msg.match(token)
                    if msg_match is not None:
                        current_cmd['from'] = msg_match.group(1)
                        current_cmd['msg'] = msg_match.group(2)
                        result.append(current_cmd)
            else:
                if cmd_match.group(1) in self.allowed_commands:
                    current_cmd = {
                        'cmd': cmd_match.group(1)
                    }
        return result