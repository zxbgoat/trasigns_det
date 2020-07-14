ARG PYTORCH="1.5.1"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt install gcc\
    && pip install scikit-image\
    && pip install pycocotools
RUN git clone https://github.com/zxbgoat/trasigns_det.git /trasigns

WORKDIR /trasigns

ENTRYPOINT python main.py
CMD /data/tt100k
