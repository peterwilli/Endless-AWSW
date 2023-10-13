from jina import Client
from docarray import DocList
from docs import InputDoc, OutputDoc

if __name__ == '__main__':
    c = Client(host='grpc://0.0.0.0:54321')
    da = c.post('/', DocList[InputDoc]([
        InputDoc(
            past = "",
            prompt = "Hey Sebastian. I am with someone also named Sebastian on VC in Discord and he has you as his profile picture and he asks you out for a date! Do you want to as well?"
        )
    ]), return_type=DocList[OutputDoc])
    for doc in da:
        print(doc)
