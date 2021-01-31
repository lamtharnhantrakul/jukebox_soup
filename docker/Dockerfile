#FROM continuumio/anaconda3

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install git
RUN apt update
RUN apt install -y git

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

RUN apt update
RUN apt install -y libopenmpi-dev

RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
