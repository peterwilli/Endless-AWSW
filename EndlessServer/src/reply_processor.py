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

    def commands_to_string(self, commands) -> str:
        result = []
        for cmd in commands:
            result_item = ""
            if cmd['cmd'] == "msg":
                result_item += f"<{cmd['cmd']}>"
                type_prefix = ""
                if cmd['from'] == "c":
                    type_prefix = "<p>"
                result_item += f"{type_prefix}{cmd['from']} \"{cmd['msg']}\""
            if cmd['cmd'] == "scn":
                # only dragons have scn so we can safely prefix a dragon reply token here
                result_item += "<d>"
                result_item += f"<{cmd['cmd']}>"
                result_item += cmd['scn']
            result.append(result_item)
        return "".join(result)

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