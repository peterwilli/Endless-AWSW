from transformers import Trainer, TrainingArguments
import re
import sys
import logging
from docarray import DocumentArray, Document

class ReplyProcessor:
    def __init__(self):
        self.re_token = re.compile(r'(<.*?>|[^<]*)')
        self.re_command = re.compile(r'^<(.*?)>$')
        self.re_msg = re.compile(r'([A-Za-z]{1,2})\s(.*?)\s{0,1}"(.*)"')

    def docs_to_string(self, docs: DocumentArray) -> str:
        result = []
        for doc in docs:
            result_item = ""
            cmd = doc.tags['cmd']
            if cmd == 'msg':
                msg_from = doc.tags['from']
                if msg_from == "c":
                    result_item += "<p>"
                result_item += f"<{cmd}>"
                result_item += f"{msg_from}"
                if 'emotion' in doc.tags and len(doc.tags['emotion']) > 0:
                    result_item += f" {doc.tags['emotion']}"
                result_item += f" \"{doc.text}\""
            elif cmd == "scn":
                # only dragons have scn so we can safely prefix a dragon reply token here
                result_item += "<d>"
                result_item += f"<{cmd}>"
                result_item += doc.text
            result.append(result_item)
        return "".join(result)

    def string_to_docs(self, string) -> DocumentArray:
        result = []
        current_doc = None
        for token in self.re_token.findall(string):
            cmd_match = self.re_command.match(token)
            if cmd_match is None:
                if current_doc.tags['cmd'] == 'scn':
                    current_doc.text = token
                    result.append(current_doc)
                elif current_doc.tags['cmd'] == 'msg':
                    msg_match = self.re_msg.match(token)
                    if msg_match is not None:
                        msg_from = msg_match.group(1)
                        current_doc.tags['from'] = msg_from
                        emotion = msg_match.group(2)
                        if emotion is not None:
                            current_doc.tags['emotion'] = emotion
                        current_doc.text = msg_match.group(3)
                        result.append(current_doc)
            else:
                current_doc = Document(tags={'cmd': cmd_match.group(1)})
        return DocumentArray(result)

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