from docarray import DocumentArray, Document
from jina import Executor, requests
from executor import EndlessAWSWExec
import json

if __name__ == "__main__":
  m = EndlessAWSWExec()
  test_past = [
      { 'cmd': 'msg', 'from': 'c', 'msg': "Hey Remy!" },
      { 'cmd': 'scn', 'scn': 'park2' },
      { 'cmd': 'msg', 'emotion': 'smile', 'from': 'Ry', 'msg': "Hey!" },
  ]
  da = DocumentArray([Document(text=json.dumps(test_past), tags={ 'prompt': 'Test' })])
  da = m.send_message(da)
  print(f'Reply: {da[0].text}')