FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04
MAINTAINER Peter Willemsen <peter@codebuffet.co>
RUN echo "Installing dependencies..." && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl git build-essential cmake python3-pip python3 pkg-config libssl-dev zsh clang && \
    apt-get dist-upgrade -y && \
	rm -rf /var/lib/apt/lists/*
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && rm /requirements.txt
WORKDIR /opt/awsw

# Add dev user
RUN useradd -ms /bin/zsh awsw-dev && \
    usermod -aG sudo awsw-dev && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER awsw-dev

CMD ["jupyter", "lab", "--ip=0.0.0.0"]