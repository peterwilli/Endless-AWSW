from docarray import DocumentArray, Document
from jina import Executor, requests
from .executor import EndlessAWSWExec
import json

if __name__ == "__main__":
  m = EndlessAWSWExec()
  test_past = DocumentArray([
      Document(text = "Hey Remy!", tags = { 'cmd': 'msg', 'from': 'c' }),
      Document(text = "park2", tags = { 'cmd': 'scn' }),
      Document(text = "Hey!", tags = { 'emotion': 'smile', 'from': 'Ry' }),
  ])
  reply = m.send_message(test_past, {
    'prompt': 'Test' 
  })
  print(f'Reply: {da[0].text}')