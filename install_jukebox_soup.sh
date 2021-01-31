#!/bin/sh

git config --global user.email "hanoi7@gmail.com"
git config --global user.name "lamtharnhantrakul"

sudo apt-get install python3-pip

/usr/bin/python3 -m pip install --upgrade pip3

pip3 install git+https://github.com/openai/jukebox.git