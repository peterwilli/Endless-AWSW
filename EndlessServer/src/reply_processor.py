from transformers import Trainer, TrainingArguments
import torch
import re
import sys
import logging

class ReplyProcessor:
    def __init__(self):
        self.end_token = "<|endoftext|>"
        self.splitter = re.compile(r'\s')
        self.allowed_characters = {
            'c': 'Player',
            'Ry': 'Remy',
            'Lo': 'Lorem',
            'Ip': 'Ipsum',
            'Br': 'Bryce',
            'Wr': 'Unknown name',
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
        reply = f"msg d {reply}"
        tokens = self.splitter.split(reply)
        from_character = None
        commands = []
        current_cmd = None

        def skip_command():
            nonlocal current_cmd
            current_cmd = None

        def save_command():
            nonlocal current_cmd
            commands.append(current_cmd)
            current_cmd = None

        for token in tokens:
            if len(token) == 0:
                continue
            if current_cmd is None:
                if token in self.allowed_commands:
                    current_cmd = {
                        'cmd': token
                    }
            elif current_cmd['cmd'] == "scn":
                if not 'scn' in current_cmd:
                    current_cmd['scn'] = token
                    save_command()
            elif current_cmd['cmd'] == "msg":
                if not 'type' in current_cmd:
                    current_cmd['type'] = token
                elif not 'from' in current_cmd:
                    if token in self.allowed_characters:
                        current_cmd['from'] = token
                        current_cmd['msg'] = []
                    else:
                        skip_command()
                        continue
                elif token.startswith("\""):
                    if token.endswith("\""):
                        # edge case for 1 word sentences
                        current_cmd['msg'].append(token[1:-1])
                        save_command()
                    else:
                        current_cmd['msg'].append(token[1:])
                elif token.endswith("\""):
                    current_cmd['msg'].append(token[:-1])
                    save_command()
            elif token == self.end_token:
                break
        return { 'cmds': commands }