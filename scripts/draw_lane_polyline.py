#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import pickle
import cv2
import glob

def create_lane_json():
    image_path_list = glob.glob("/Users/take/fun/dataset/bdd100k/labels/lane/masks/train/*.png")

    if os.path.exists("./lane_ij.json"):
        os.remove("./lane_ij.json")

    f = open("./lane_ij.json", mode="w")

    debug_cnt = 0
    for image_path in image_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        j_size = img.shape[0]
        i_size = img.shape[1]

        # print("height:{}, width:{}".format(j_size, i_size))
        print(image_path)

        ij_dic = {"lane":[], "image":os.path.basename(image_path)}
        for j in range(j_size):
            for i in range(i_size):
                pix = img[j, i]
                if pix < 255:
                    ij_dic["lane"].append([j, i])
        f.write(str(ij_dic).replace("'", '"') + "\n")
        debug_cnt += 1
        if debug_cnt > 5:
            break
    f.close()

if __name__ == "__main__":
    create_lane_json()

    f = open("./lane_ij.json", "r")
    for l in f:
        print(l)



