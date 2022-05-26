#!/bin/bash

set -e

while getopts w:a:e: flag
do
    case "${flag}" in
        w) WANDB=${OPTARG};;
        a) ALGO=${OPTARG};;
        e) ENV=${OPTARG};;
    esac
done

sudo apt install swig screen
pip install -r requirements.txt
pip install atari-py timm

tar -xf atari-roms.tar.gz
python -m atari_py.import_roms ROMS
rm -f atari-roms.tar.gz
rm -rf ROMS

screen -d -m bash -c "WANDB_API_KEY=$WANDB python train.py --algo $ALGO --env $ENV \
					--vec-env subproc --track --wandb-project-name $ALGO-$ENV \
					--wandb-entity yeeeb"