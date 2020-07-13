FROM pytorch/pytorch
RUN apt install gcc\
    && pip install scikit-image\
    && pip install pycocotools
