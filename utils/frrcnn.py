import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def build_model(clsnum, minsz, maxsz):
    frrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    frrcnn.transform = GeneralizedRCNNTransform(min_size=minsz, max_size=maxsz,
                                                image_mean=[0.485, 0.456, 0.406],
                                                image_std=[0.229, 0.224, 0.225])
    in_features = frrcnn.roi_heads.box_predictor.cls_score.in_features
    frrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, cls_num)
    return frrcnn
