#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import ast
import cv2
import torch
import numpy as np
from torch.utils import  data
from data_augumentation import *

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train":Compose([
                ConvertFromInts(), ToAbsoluteCoords(),
                PhotometricDistort(), Expand(color_mean),
                RandomSampleCrop(), RandomMirror(),
                ToPercentCoords(), Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            "val":Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)

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
                if len(anno["bbox"]) > 0:
                    boxes = []
                    # normalized bboxs
                    for bbox in anno["bbox"]:
                        boxes.append([(bbox[0] / width),
                                      (bbox[1] / height),
                                      (bbox[2] / width),
                                      (bbox[3] / height), 1]) # label_idx car==1

                    boxes = np.array(boxes, dtype=np.float32)
                    if self.transform:
                        img, _boxes, labels = self.transform(img, self.phase, boxes[:, :4], boxes[:, 4])
                        gt = np.hstack((_boxes, np.expand_dims(labels, axis=1)))
                    else:
                        gt = boxes
                else:
                    gt = np.empty(0, dtype=np.float32)

                img = torch.from_numpy(img[:,:,(2, 1, 0)]).permute(2, 0, 1) # convert BGR->RGB and (H, W, C) to (C, H, W)
                return img, gt

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
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)
    #dataset = SSDDataset(anno_list, phase="train",transform=transform)
    dataset = SSDDataset(anno_list, phase="train",transform=None)
    img, bboxs = dataset.__getitem__(5)
    print(bboxs)

    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=cusdom_collate_fn)

    batch_iter = iter(dataloader)
    imgs, targets = next(batch_iter)
    print(imgs.size())
    print(len(targets))
