import torch
import argparse
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--impath', type=str, default='imgs/2.jpg',
                    help='The path of the image to be tested')
parser.add_argument('--wtpath', type=str,
                    default='weights/fasterrcnn_r50_fpn_10e_tt100k_2048px.pkl',
                    help='The weights of the model to be tested')
parser.add_argument('--rspath', type=str, default='imgs/rst.jpg',
                    help='The path of the file which save the result')
args = parser.parse_args()

device = torch.device('cuda:2')
model = torch.load(args.wtpath)
model.eval().to(device)
img = Image.open(args.impath)
img = F.to_tensor(img).to(device)
rst = model([img])[0]
boxes = rst['boxes'].cpu().detach().numpy()
labels = rst['labels'].cpu().detach().numpy()
scores = rst['scores'].cpu().detach().numpy()
print(boxes)
print(scores)
print(labels)

img = Image.open(args.impath)
print(img)
draw = ImageDraw.Draw(img)
print(draw)
for i in range(len(boxes)):
    score = scores[i]
    if score <= 0.5:
        continue
    label = labels[i]
    box = boxes[i]
    draw.rectangle(box, width=3)

rspath = args.rspath
img.save(rspath)
