#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import cv2
import glob

if __name__ == "__main__":
    image_path_list = glob.glob("/Users/take/fun/dataset/bdd100k/labels/lane/masks/train/*.png")

    for image_path in image_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        j_size = img.shape[0]
        i_size = img.shape[1]

        ij_list = []
        for j in range(j_size):
            for i in range(i_size):
                pix = img[j, i]
                if pix < 255:
                    ij_list.append([j, i])
        print(ij_list)
        break

    pass
