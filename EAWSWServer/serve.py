from executor import EndlessAWSWExec
from jina import Flow, Executor, requests

with Flow(protocol='HTTP', port=5000).add(name='eawsw', uses=EndlessAWSWExec) as f:
    f.block()