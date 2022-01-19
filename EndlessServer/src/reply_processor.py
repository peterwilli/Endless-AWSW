from transformers import Trainer, TrainingArguments
import re
import sys
import Levenshtein
import logging

class ReplyProcessor:
    def __init__(self):
        self.re_token = re.compile(r'(<.*?>|[^<]*)')
        self.re_command = re.compile(r'^<(.*?)>$')
        self.re_msg = re.compile(r'([a-zA-Z]{1,2})\s"(.*?)"')
        self.re_brackets = re.compile(r'\[(.*?)]')
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
        self.allowed_scenes = ['park2', 'black', 'loremapt', 'office', 'bare', 'bareblur', 'bareblur2', 'pad', 'facin2', 'facinx', 'facin3', 'alley', 'farm', 'town4', 'beach', 'adineapt', 'corridor', 'emeraroom', 'o4', 'park3', 'np3x', 'np2x', 'np1x', 'buildingoutside', 'o2', 'np3', 'np2', 'store2', 'town1x', 'forestx', 'cave', 'o', 'remyapt', 'cafe', 'viewingspot', 'np1r', 'hallway', 'np2y', 'np1n', 'town2', 'stairs', 'darker', 'town1', 'store', 'library', 'school', 'forest1', 'forest2', 'storex', 'np5e', 'port1', 'beachx', 'padx', 'intro1', 'intro2', 'np4', 'np5', 'fac1', 'facin', 'town3', 'kitchen', 'np1', 'stars', 'o3', 'town7', 'town6', 'deadbody', 'whiteroom', 'office2', 'cave2', 'table', 'starsrx', 'hatchery', 'farm2', 'gate', 'testingroom', 'np6', 'fac12', 'adineapt2']
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

    def has_unclosed_or_nested_brackets(self, text) -> bool:
        is_ok = True
        for char in text:
            if char == '[':
                if is_ok:
                    is_ok = False
                else:
                    return True
            elif char == ']':
                if is_ok:
                    return True
                else:
                    is_ok = True
        return not is_ok

    def has_valid_bracket_vars(self, text) -> bool:
        valid_var_names = ['player_name']

        for var_name in self.re_brackets.findall(text):
            if var_name not in valid_var_names:
                return False
                
        return True

    def post_process_reply(self, reply):
        result = []
        current_cmd = None
        for token in self.re_token.findall(reply):
            cmd_match = self.re_command.match(token)
            if cmd_match is None:
                if current_cmd['cmd'] == 'scn':
                    if not token in self.allowed_scenes:
                        return None
                    current_cmd['scn'] = token
                    result.append(current_cmd)
                elif current_cmd['cmd'] == 'msg':
                    msg_match = self.re_msg.match(token)
                    if msg_match is not None:
                        msg_from = msg_match.group(1)
                        if not msg_from in self.allowed_characters:
                            return None
                        if msg_from == 'c':
                            # From player, we end if a dragon has been before us,
                            # otherwise we see it as invalid
                            has_old_msg = False
                            for old_cmd in result:
                                if old_cmd['cmd'] == 'msg':
                                    has_old_msg = True
                                    break
                            if has_old_msg:
                                return result
                            else:
                                return None
                        current_cmd['from'] = msg_from
                        current_cmd['msg'] = msg_match.group(2)
                        if self.has_unclosed_or_nested_brackets(current_cmd['msg']):
                            return None
                        if not self.has_valid_bracket_vars(current_cmd['msg']):
                            return None
                        for old_cmd in result:
                            if old_cmd['cmd'] == 'msg':
                                ratio = Levenshtein.ratio(current_cmd['msg'], old_cmd['msg'])
                                logging.debug(f"ratio: {ratio} {current_cmd['msg']} <> {old_cmd['msg']}")
                                if ratio > 0.7:
                                    return None
                        result.append(current_cmd)
            else:
                if cmd_match.group(1) in self.allowed_commands:
                    current_cmd = {
                        'cmd': cmd_match.group(1)
                    }
        return result