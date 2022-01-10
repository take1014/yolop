#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import json
import cv2
import numpy as np
import glob
import ast

def draw_lane_line(img, points):
    print("points count:{}".format(len(points)))
    if len(points) > 0:
        for ps in points:
            poly = np.array(ps["poly"], dtype=np.int32)
            lanedir = ps["laneDir"]
            for i in range(len(poly)):
                if i+1 < len(poly):
                    p1 = poly[i]
                    p2 = poly[i+1]
                    # print(p1, p2, lanetype)
                    if lanedir == "vertical":
                        # vertical
                        cv2.line(img, p1, p2, color=(255, 0, 0), thickness=2)
                    else:
                        # parallel
                        cv2.line(img, p1, p2, color=(0, 255, 0), thickness=2)
                    cv2.circle(img, p1, 3, color=(0, 0, 255), thickness=2)
                    cv2.circle(img, p2, 3, color=(0, 0, 255), thickness=2)
    return img

if __name__ == "__main__":
    lane_file_list = glob.glob("/Users/take/fun/dataset/bdd100k/json/lane/train/*.json")

    for lane_file in lane_file_list:
        img = None
        with open(lane_file, "r") as f:
            for l in f:
                dict_info = ast.literal_eval(l)
                print(dict_info)
                points = dict_info["lane"]
                image_file_path = os.path.join("/Users/take/fun/dataset/bdd100k/images/100k/train", dict_info["raw_file"])
                img = cv2.imread(image_file_path)
                print(img.shape[0], img.shape[1])
                img = draw_lane_line(img, points)

        bbox_file_path = "/Users/take/fun/dataset/bdd100k/json/bbox/train/{}".format(os.path.basename(lane_file))
        if os.path.exists(bbox_file_path):
            with open(bbox_file_path, "r") as f:
                for l in f:
                    dict_info = ast.literal_eval(l)
                    bboxs = dict_info["bbox"]
                    for bbox in bboxs:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[2]), int(bbox[3]))
                        print(p1, p2)
                        cv2.rectangle(img, p1, p2, (0, 255, 255), thickness=2)
        cv2.imshow("image", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()



    # for i in range(lane_manage.get_data_len()):
    #     points, image_name = lane_manage.get_points(i)
    #     image_file_path = os.path.join("/Users/take/fun/dataset/bdd100k/images/100k/train", image_name)
    #     img = cv2.imread(image_file_path)
    #     print(img.shape[0], img.shape[1])
    #     img = draw_lane_line(img, points)
    #     cv2.imshow("image", img)
    #     cv2.waitKey(1000)
    #     cv2.destroyAllWindows()
