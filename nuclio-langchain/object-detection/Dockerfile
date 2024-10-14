FROM ultralytics/ultralytics

RUN sed -i 's|http://archive.ubuntu.com/|http://mirrors.ustc.edu.cn/|g' /etc/apt/sources.list

RUN apt-get update && apt-get -y install openssh-client libgl1-mesa-glx libglib2.0-dev
RUN mkdir -p /root/.pip \
        && echo "[global]" > /root/.pip/pip.conf \
        && echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf

WORKDIR /opt/nuclio

COPY model.pt /opt/nuclio/model.pt
COPY function.yaml /opt/nuclio/function.yaml