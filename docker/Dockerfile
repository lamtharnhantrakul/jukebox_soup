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

RUN pip install torch torchvision
