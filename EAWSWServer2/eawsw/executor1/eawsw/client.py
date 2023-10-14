from jina import Client
from docarray import DocList
from docs import InputDoc, ReplyDoc, CommandDoc

if __name__ == '__main__':
    c = Client(host='grpc://0.0.0.0:54321')
    da = c.post('/send_message', DocList[InputDoc]([
        InputDoc(
            prompt = "Hey Sebastian. I am with someone also named Sebastian on VC in Discord and he has you as his profile picture and he asks you out for a date! Do you want to as well?"
        )
    ]), return_type=DocList[ReplyDoc])

    for doc in da:
        for cmd in doc.commands:
            print(cmd)
