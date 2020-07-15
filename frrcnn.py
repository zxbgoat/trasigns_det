import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


cls_num = 83
frrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = frrcnn.roi_heads.box_predictor.cls_score.in_features
frrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, cls_num)
