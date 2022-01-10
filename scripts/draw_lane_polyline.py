#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import json
import cv2
import numpy as np
from data_manager import LaneDataManager

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
    lane_manage = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")

    for i in range(lane_manage.get_data_len()):
        points, image_name = lane_manage.get_points(i)
        image_file_path = os.path.join("/Users/take/fun/dataset/bdd100k/images/100k/train", image_name)
        img = cv2.imread(image_file_path)
        print(img.shape[0], img.shape[1])
        img = draw_lane_line(img, points)
        cv2.imshow("image", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
