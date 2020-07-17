import json
import torch
import argparse
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms import functional as F


categories = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14',
              'i2', 'i3', 'i4', 'i5', 'il', 'io', 'ip',
              'p1', 'p10', 'p11', 'p12', 'p13', 'p14',
              'p15', 'p16', 'p17', 'p18', 'p19', 'p2',
              'p20', 'p21', 'p22', 'p23', 'p24', 'p25',
              'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6',
              'p8', 'p9', 'pa', 'pb', 'pg', 'ph', 'pl',
              'pm', 'pn', 'pne', 'po', 'pr', 'ps', 'pw',
              'w10', 'w12', 'w13', 'w15', 'w16', 'w18',
              'w20', 'w21', 'w22', 'w3', 'w30', 'w32',
              'w34', 'w35', 'w37', 'w38', 'w41', 'w42',
              'w45', 'w46', 'w47', 'w5', 'w55', 'w57',
              'w58', 'w59', 'w63', 'w66', 'w8', 'wo']

describs = {'i1': u'步行', 'i2': u'非机动车形式行驶', 'i3': u'环岛行驶',
            'i4': u'机动车行驶', 'i5': u'靠右侧道路行驶', 'i6': u'靠左侧道路行驶',
            'i7': u'立交直行和右转弯行驶', 'i8': u'立交直行和左转弯行驶', 'i9': u'鸣喇叭',
            'i10': u'向右转弯', 'i11': u'向左和向右转弯', 'i12': u'向左转弯',
            'i13': u'直行', 'i14': u'直行和向右转弯', 'i15': u'直行和向左转弯',
            'il': u'最低限速', 'ip': u'人行横道',
            'p1': u'禁止超车', 'p2': u'禁止畜力车进入', 'p3': u'禁止大型客车驶入',
            'p5': u'禁止掉头', 'p6': u'禁止非机动车进入', 'p8': u'禁止汽车拖、挂车驶入',
            'p9': u'禁止行人进入', 'p10': u'禁止机动车驶入', 'p11': u'禁止鸣喇叭',
            'p12': u'禁止二轮摩托车驶入', 'p13': u'禁止这两种车驶入', 'p14': u'禁止直行',
            'p15': u'禁止人力车进入', 'p16': u'禁止人力货运三轮车进入', 'p17': u'禁止人力客运三轮车进入',
            'p18': u'禁止三轮车机动车通行'}


def infer(impath, svrst=True, svpath='imgs/rst.jpg', dev='cuda:2',
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
            des =  categories[label-1]
            box = boxes[i]
            draw.rectangle(box, width=3)
            x, y = box[0], box[1]
            draw.text([x-10, y-10], des, fill=(255,0,255))
        img.save(svpath)

    return rsts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('impath', type=str,
                        help='The path of the image to be tested')
    parser.add_argument('--wtpath', type=str,
                        default='weights/fasterrcnn_r50_fpn_10e_tt100k_2048px.pkl',
                        help='The weights of the model to be tested')
    parser.add_argument('--svpath', type=str, default='imgs/rst.jpg',
                        help='The path of the file which save the result')
    args = parser.parse_args()
    infer(args.impath)
