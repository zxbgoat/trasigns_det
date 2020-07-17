import json
import torch
import argparse
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms import functional as F


def infer(impath, svrst=False, svpath='rst.jpg', dev='cuda:2',
          wtpath='weights/fasterrcnn_r50_fpn_10e_tt100k_2048px.pkl'):
    device = torch.device(dev)
    model = torch.load(wtpath)
    model.eval().to(device)
    img = Image.open(impath)
    img = F.to_tensor(img).to(device)
    ret = model([img])[0]
    boxes = ret['boxes'].cpu().detach().numpy().tolist()
    labels = ret['labels'].cpu().detach().numpy().tolist()
    scores = ret['scores'].cpu().detach().numpy().tolist()
    rsts = []
    for i in range(len(scores)):
        rst = {}
        rst['label'] = labels[i]
        rst['score'] = scores[i]
        rst['box'] = boxes[i]
        rsts.append(rst)
    rsts = json.dumps(rsts)
    print(rsts)

    if svrst:
        img = Image.open(impath)
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
        img.save(svpath)

    return rsts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--impath', type=str, default='imgs/2.jpg',
                        help='The path of the image to be tested')
    parser.add_argument('--wtpath', type=str,
                        default='weights/fasterrcnn_r50_fpn_10e_tt100k_2048px.pkl',
                        help='The weights of the model to be tested')
    parser.add_argument('--rspath', type=str, default='imgs/rst.jpg',
                        help='The path of the file which save the result')
    args = parser.parse_args()
    infer(args.impath)
