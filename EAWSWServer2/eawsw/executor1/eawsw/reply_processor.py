import re
import sys
import logging
from docarray import DocList
from .docs import ReplyDoc, CommandDoc

class ReplyProcessor:
    def __init__(self):
        self.re_token = re.compile(r'(<.*?>|[^<]*)')
        self.re_command = re.compile(r'^<(.*?)>$')
        self.re_msg = re.compile(r'([A-Za-z]{1,2})\s(.*?)\s{0,1}"(.*)"')

    def docs_to_string(self, docs: DocList[CommandDoc]) -> str:
        result = []
        for doc in docs:
            result_item = ""
            cmd = doc.cmd
            if cmd == 'msg':
                msg_from = doc.msg_from
                if msg_from == "c":
                    result_item += "<p>"
                result_item += f"<{cmd}>"
                result_item += f"{msg_from}"
                if doc.emotion is not None:
                    result_item += f" {doc.emotion}"
                result_item += f" \"{doc.value}\""
            elif cmd == "scn":
                # only dragons have scn so we can safely prefix a dragon reply token here
                result_item += "<d>"
                result_item += f"<{cmd}>"
                result_item += doc.value
            result.append(result_item)
        return "".join(result)

    def string_to_docs(self, string) -> ReplyDoc:
        result = ReplyDoc(commands=DocList[CommandDoc]())
        current_doc = None
        for token in self.re_token.findall(string):
            cmd_match = self.re_command.match(token)
            if cmd_match is None:
                if current_doc.cmd == 'scn':
                    current_doc.value = token
                    result.commands.append(current_doc)
                elif current_doc.cmd == 'msg':
                    msg_match = self.re_msg.match(token)
                    if msg_match is not None:
                        msg_from = msg_match.group(1)
                        current_doc.msg_from = msg_from
                        emotion = msg_match.group(2)
                        if emotion is not None and len(emotion) > 0:
                            current_doc.emotion = emotion
                        current_doc.value = msg_match.group(3)
                        result.commands.append(current_doc)
            else:
                current_doc = CommandDoc(cmd=cmd_match.group(1))
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_tokens = '<p><msg>c "Flooding?"<d><scn>o2<msg>Sb "Yes."'
    rp = ReplyProcessor()
    docs = rp.string_to_docs(test_tokens)
    print(docs.to_json())
    raw = rp.docs_to_string(docs.commands)
    if test_tokens == raw:
        logging.info("testing ok")
    else:
        logging.error(f"test_tokens != cmd_raw\n{test_tokens}\n{raw}")