#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import ast
import cv2
import torch
import numpy as np
from torch.utils import  data

class SSDDataset(data.Dataset):
    def __init__(self, anno_list, phase, transform=None):
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):
        with open(self.anno_list[idx], mode="r") as f:
            for l in f:
                anno = ast.literal_eval(l)
                img = cv2.imread("/Users/take/fun/dataset/bdd100k/images/100k/train/"+anno["raw_file"])
                height, width, channels = img.shape
                bboxs = np.array(anno["bbox"], dtype=np.float32)
                # normalized bboxs
                for bbox in bboxs:
                    bbox[0] /= width
                    bbox[1] /= height
                    bbox[2] /= width
                    bbox[3] /= height
                if self.transform:
                    img, bboxs = self.transform(img, self.phase, bboxs)
                img = torch.from_numpy(img[:,:,(2, 1, 0)]).permute(2, 0, 1) # convert BGR->RGB and (H, W, C) to (C, H, W)
                return img, torch.from_numpy(bboxs)

def cusdom_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets

if __name__ == "__main__":
    import glob
    anno_list = glob.glob("/Users/take/fun/dataset/bdd100k/json/bbox/train/*.json")
    dataset = SSDDataset(anno_list, phase="train")
    img, bboxs = dataset.__getitem__(5)
    print(type(img))
    print(type(bboxs))


    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=cusdom_collate_fn)

    batch_iter = iter(dataloader)
    imgs, targets = next(batch_iter)
    print(imgs.size())
    print(len(targets))
