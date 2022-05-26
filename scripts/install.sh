#!/bin/bash

sudo apt install swig
pip install -r requirements.txt

tar -xf atari-roms.tar.gz
python -m atari_py.import_roms ROMS
rm -f atari-roms.tar.gz
rm -rf ROMS