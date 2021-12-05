FROM python:3.7.12-slim-bullseye
MAINTAINER Peter Willemsen <peter@codebuffet.co>
# Cache this one as it's like 2 GB...
RUN pip install torch
RUN pip install Levenshtein flask-restful transformers
ENV FLASK_ENV=development
WORKDIR /opt/awsw
CMD ["python", "-u", "main.py"]