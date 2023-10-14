from jina import Executor, requests
from docarray import DocList
from .reply_processor import ReplyProcessor
from .docs import InputDoc, ReplyDoc, CommandDoc
from .model_manager import ModelManager
from .bad_prompt_filter import is_disallowed
import sys

filter_profanity = True
model_manager = ModelManager("/Projects/Personal/Endless-AWSW/Research/merged-eawsw-16k")
reply_processor = ReplyProcessor()
# model_manager = ModelManager("peterwilli/eawsw-16k")
command_retries = 5

class EndlessAWSWExec(Executor):
    @requests(on=['/send_message'])
    async def send_message(self, docs: DocList[InputDoc], **kwargs) -> DocList[ReplyDoc]:
        result = []
        for doc in docs:
            past_text = reply_processor.docs_to_string(doc.past)
            if is_disallowed(doc.prompt):
                return DocList[ReplyDoc]([
                    ReplyDoc(
                        commands=DocList[CommandDoc]([
                            CommandDoc(cmd="error", value="profanity_detected")
                        ])
                    )
                ])

            found_reply = False
            for i in range(command_retries):
                reply = await model_manager.say(past_text, doc.prompt)
                if not is_disallowed(reply):
                    cmd_docs = reply_processor.string_to_docs(reply)
                    result.append(ReplyDoc(commands=cmd_docs))
                    found_reply = True
                    break
            if not found_reply:
                result.append(ReplyDoc(commands=[]))
        return DocList[ReplyDoc](result)
