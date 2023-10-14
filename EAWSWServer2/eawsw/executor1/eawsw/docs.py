from docarray import BaseDoc, DocList
from typing import Optional

class CommandDoc(BaseDoc):
    cmd: str
    emotion: Optional[str]
    msg_from: Optional[str]
    value: Optional[str]

class ReplyDoc(BaseDoc):
    commands: DocList[CommandDoc]

class InputDoc(BaseDoc):
    past: Optional[DocList[CommandDoc]] = DocList[CommandDoc]([])
    prompt: str
    