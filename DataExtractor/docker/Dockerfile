FROM debian:buster-slim
MAINTAINER Peter Willemsen <peter@codebuffet.co>

RUN echo "Installing dependencies..." && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y zsh curl nano python-pip git renpy && \ 
    DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y && \
	rm -rf /var/lib/apt/lists/*

RUN pip2 install -U cython future six

WORKDIR /src
CMD ["sleep", "infinity"]