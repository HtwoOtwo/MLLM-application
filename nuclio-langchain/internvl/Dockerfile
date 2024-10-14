FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get -y install curl git python3 python3-pip
RUN apt-get update && apt-get install -y openssh-client libgl1-mesa-glx libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip \
        && echo "[global]" > /root/.pip/pip.conf \
        && echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf
    

# RUN MAX_JOBS=4 pip3 install flash-attn --no-build-isolation

RUN pip3 install timm
RUN pip3 install transformers
RUN pip3 install opencv-python termcolor yacs pyyaml scipy pynvml
RUN pip3 install deepspeed
RUN pip3 install pycocoevalcap tqdm 

RUN pip3 install accelerate sentencepiece einops

# download model locally to avoid invoke-time downloading
WORKDIR /opt/nuclio
COPY download_model.py /opt/nuclio/download_model.py
RUN python3 download_model.py
