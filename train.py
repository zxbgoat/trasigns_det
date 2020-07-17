import torch
import utils
import argparse
import torchvision
from data.tt100k import TT100K
from data.transforms import get_transform
import utils
from utils.frrcnn import build_model
from utils.engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--dtdir', type=str,
                    default='/home/tesla/Workspace/dataset/tt100k',
                    help='directory path of the dataset')
parser.add_argument('--clsnum', type=int, default=83,
                    help='class num of the model')
parser.add_argument('--minsz', type=int, default=2048,
                    help='min size of the image while training')
parser.add_argument('--maxsz', type=int, default=2048,
                    help='max size of the image while training')
parser.add_argument('--bchsz', type=int, default=1,
                    help='batch size while training')
parser.add_argument('--epcnum', type=int, default=10,
                    help='epoch num to be trained')
args = parser.parse()
print(args)

dataset = TT100K(args.dtdir, transforms=get_transform(train=True))
dataset_test = TT100K(args.dtdir, split='test', transforms=get_transform(train=False))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.bchsz, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.device_count() > 1:
    frrcnn = torch.nn.DataParallel(frrcnn)
frrcnn.to(device)
params = [p for p in frrcnn.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

for epoch in range(args.epcnum):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(frrcnn, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(frrcnn, data_loader_test, device=device)

modelpath = 'fasterrcnn_r50_fpn_10e_tt100k.pkl'
torch.save(frrcnn, modelpath)
