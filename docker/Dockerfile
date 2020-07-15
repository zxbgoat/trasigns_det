ARG PYTORCH="1.5.1"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt-get update \
    && apt-get install -y gcc \
    && apt-get install -y git \
    && apt-get install -y tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools
RUN git clone https://github.com/zxbgoat/trasigns_det.git /trasigns

WORKDIR /trasigns

ENTRYPOINT python main.py
CMD /data/tt100k
