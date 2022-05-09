from transformers import Trainer, TrainingArguments
import re
import sys
import logging

class ReplyProcessor:
    def __init__(self):
        self.re_token = re.compile(r'(<.*?>|[^<]*)')
        self.re_command = re.compile(r'^<(.*?)>$')
        self.re_msg = re.compile(r'([A-Za-z]{1,2})\s(.*?)"(.*)"')

    def commands_to_string(self, commands) -> str:
        result = []
        for cmd in commands:
            result_item = ""
            if cmd['cmd'] == "msg":
                if cmd['from'] == "c":
                    result_item += "<p>"
                result_item += f"<{cmd['cmd']}>"
                result_item += f"{cmd['from']} \"{cmd['msg']}\""
            if cmd['cmd'] == "scn":
                # only dragons have scn so we can safely prefix a dragon reply token here
                result_item += "<d>"
                result_item += f"<{cmd['cmd']}>"
                result_item += cmd['scn']
            result.append(result_item)
        return "".join(result)

    def string_to_commands(self, reply):
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
                        msg_from = msg_match.group(1)
                        current_cmd['from'] = msg_from
                        current_cmd['msg'] = msg_match.group(3)
                        result.append(current_cmd)
            else:
                current_cmd = {
                    'cmd': cmd_match.group(1)
                }
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_tokens = '<p><msg>c "Flooding?"<d><scn>o2<msg>Sb "Yes."'
    rp = ReplyProcessor()
    cmd_json = rp.post_process_reply(test_tokens)
    cmd_raw = rp.commands_to_string(cmd_json)
    if test_tokens == cmd_raw:
        logging.info("testing ok")
    else:
        logging.error(f"test_tokens != cmd_raw\n{test_tokens}\n{cmd_raw}")