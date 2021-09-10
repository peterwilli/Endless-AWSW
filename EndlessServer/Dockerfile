FROM python:3.7.12-slim-bullseye
MAINTAINER Peter Willemsen <peter@codebuffet.co>
RUN pip install transformers torch flask-restful
ENV FLASK_ENV=development
WORKDIR /opt/awsw
CMD ["python", "-u", "main.py"]