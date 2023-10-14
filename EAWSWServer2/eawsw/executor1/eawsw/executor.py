from jina import Executor, requests
from docarray import DocList
from .reply_processor import ReplyProcessor
from .docs import InputDoc, OutputDoc, ReplyDoc
from .model_manager import ModelManager
import sys

filter_profanity = True
model_manager = ModelManager("/Projects/Personal/Endless-AWSW/Research/merged-eawsw-16k")
reply_processor = ReplyProcessor()
# model_manager = ModelManager("peterwilli/eawsw-16k")
command_retries = 5

def text_is_unsafe(text) -> bool:
    # Some words trigger false-negatives like 'penis'.
    # Instead of retraining the whole model I opted for adding them myself.
    # Ipsum why did you make me do this!?! Whyyyyyyy!
    block_words = [
        'penis'
    ]
    for word in block_words:
        if word in text:
            return True

    # Some words trigger false-positives like 'eat', 'lame' etc.
    # Instead of retraining the whole model I opted for simply ignoring them.
    # See https://gitlab.com/dimitrios/alt-profanity-check/-/issues/12
    filter_words = [
        'eat',
        'lame',
        'loser',
        'idiot',
        'shut'
    ]
    for word in filter_words:
        text = text.replace(word, '')
    text: str

class EndlessAWSWExec(Executor):
    @requests
    async def send_message(self, docs: DocList[InputDoc], **kwargs) -> DocList[ReplyDoc]:
        result = []
        for doc in docs:
            reply = await model_manager.say(doc.past, doc.prompt)
            docs = reply_processor.string_to_docs(reply)
            result.append(docs)
        return DocList[ReplyDoc](result)
