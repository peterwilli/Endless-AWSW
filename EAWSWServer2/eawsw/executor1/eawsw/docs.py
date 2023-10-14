from docarray import BaseDoc, DocList
from typing import Optional

class InputDoc(BaseDoc):
    past: str
    prompt: str

class OutputDoc(BaseDoc):
    reply: str

class CommandDoc(BaseDoc):
    cmd: str
    emotion: Optional[str]
    msg_from: Optional[str]
    value: Optional[str]

class ReplyDoc(BaseDoc):
    commands: DocList[CommandDoc]