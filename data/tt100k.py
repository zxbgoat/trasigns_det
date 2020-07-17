import os
import json
import torch
import os.path as osp
import skimage.io as sio


class TT100K:
    repeats = ['il', 'pa', 'ph', 'pl', 'pm', 'pr', 'pw']
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
    
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        annpath = osp.join(root, 'annotations.json')
        with open(annpath, 'r') as fp:
            anndict = json.load(fp)['imgs']
        self.anns = []
        if split == 'train':
            for key, ann in anndict.items():
                path = ann['path']
                if path.startswith(split):
                    self.anns.append(ann)
        else:
            for key, ann in anndict.items():
                path = ann['path']
                if not path.startswith(split):
                    continue
                obj_num = 0
                for obj in ann['objects']:
                    cat = obj['category']
                    if cat[:2] in self.repeats:
                        cat = cat[:2]
                    if cat in self.categories:
                        obj_num += 1
                if obj_num:
                    self.anns.append(ann)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        ann = self.anns[idx]
        # 读取图像
        path = ann['path']
        path = osp.join(self.root, path)
        img = sio.imread(path)
        # 读取标注
        boxes, labels, area = [], [], []
        obj_num = 0
        for obj in ann['objects']:
            cat = obj['category']
            if cat[:2] in self.repeats:
                cat = cat[:2]
            if not cat in self.categories:
                continue
            obj_num += 1
            labels.append(self.categories.index(cat) + 1)
            bbox = obj['bbox']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((xmax-xmin) * (ymax-ymin))
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['image_id'] = torch.as_tensor([ann['id']], dtype=torch.int64)
        target['iscrowd'] = torch.zeros((obj_num,), dtype=torch.int64)
        # 数据增强
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target
