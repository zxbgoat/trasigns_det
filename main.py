import sys
import torch
import utils
import torchvision
import transforms as T
from tt100k import TT100K
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


cls_num = 83
frrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = frrcnn.roi_heads.box_predictor.cls_score.in_features
frrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, cls_num)

if len(sys.argv) >= 2:
    dtdir = sys.argv[2]
else:
    dtdir = '/home/tesla/Workspace/dataset/tt100k'
dataset = TT100K(dtdir, transforms=get_transform(train=True))
dataset_test = TT100K(dtdir, split='test', transforms=get_transform(train=False))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
frrcnn.to(device)
params = [p for p in frrcnn.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 25
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(frrcnn, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(frrcnn, data_loader_test, device=device)

wtpath = 'fasterrcnn_r50_fpn_10e_tt100k.pth'
torch.save(frrcnn.state_dict(), wtpath)
modelpath = 'fasterrcnn_r50_fpn_10e_tt100k.pkl'
torch.save(frrcnn, modelpath)
