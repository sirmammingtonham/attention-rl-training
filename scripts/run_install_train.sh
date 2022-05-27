#!/bin/bash

set -e

while getopts :w:a:e: flag
do
    case "${flag}" in
        w) WANDB=${OPTARG}
            ;;
        a) ALGO=${OPTARG}
            ;;
        e) ENVS=${OPTARG}
            ;;
        ?) echo "script usage: $(basename $0) [-WANDB wandb_api_key] [-ALGO rl_algorithm] [-ENVS list_of_envs]" >&2; exit 1;;
        :) echo "Missing option argument for -$OPTARG" >&2; exit 1;;
        *) echo "Unimplemented option: -$OPTARG" >&2; exit 1;;
    esac
done

if ((OPTIND < 7))
then
    echo "script usage: $(basename $0) [-WANDB wandb_api_key] [-ALGO rl_algorithm] [-ENVS list_of_envs]"
    exit 1
fi

CMD=""

for ENV in $ENVS
do
    CMD+="WANDB_API_KEY=$WANDB python train.py --algo $ALGO --env $ENV \
					--vec-env subproc --track --wandb-project-name $ALGO-$ENV;"
done

git reset --hard HEAD
sudo apt install swig screen
pip install -r requirements.txt
pip install atari-py timm

if [ -f "atari-roms.tar.gz" ]; then
    tar -xf atari-roms.tar.gz
    python -m atari_py.import_roms ROMS
    rm -f atari-roms.tar.gz
    rm -rf ROMS
fi

screen -d -m bash -c "$CMD"