ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    wget \
    unrar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /home/code/rl_zoo/
ENV VENV /root/venv
COPY requirements.txt /tmp/


RUN \
<<<<<<< HEAD
    mkdir -p ${CODE_DIR}/rl_zoo3 && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install pip install highway-env==1.5 && \
=======
    mkdir -p ${CODE_DIR} && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install highway-env==1.5.0 && \
    pip install timm && \
    pip install atari-py && \
>>>>>>> 14e5ce7 (update dockerfile for compute instance)
    rm -rf $HOME/.cache/pip

COPY hyperparams/ ${CODE_DIR}/hyperparams/
COPY scripts/ ${CODE_DIR}/scripts/
COPY utils/ ${CODE_DIR}/utils/
COPY train.py ${CODE_DIR}


RUN \
    wget http://www.atarimania.com/roms/Roms.rar -O /tmp/roms.rar && \
    unrar e -y /tmp/roms.rar /tmp/ROMS/ && \
    python3 -m atari_py.import_roms /tmp/ROMS && \
    rm -f /tmp/roms.rar && \
    rm -rf /tmp/ROMS

ENV PATH=$VENV/bin:$PATH

# COPY docker/entrypoint.sh /tmp/
# RUN chmod +x /tmp/entrypoint.sh
# ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
