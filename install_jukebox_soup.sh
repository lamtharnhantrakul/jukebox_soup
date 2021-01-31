#!/bin/sh

git config --global user.email "hanoi7@gmail.com"
git config --global user.name "lamtharnhantrakul"

/usr/bin/python3 -m pip install --upgrade pip
sudo apt-get -y install python3-pip

pip install git+https://github.com/openai/jukebox.git

sudo apt-get update -y
sudo apt-get install -y libsndfile-dev

pip install torch torchvision
