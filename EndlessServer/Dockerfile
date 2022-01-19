FROM python:3.7.12-slim-bullseye
MAINTAINER Peter Willemsen <peter@codebuffet.co>
RUN pip install Levenshtein onnxruntime flask-restful transformers
WORKDIR /opt/awsw
CMD ["python", "-u", "main.py"]
