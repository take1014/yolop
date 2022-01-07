#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import json
import cv2
import numpy as np
from lane_data_manage import LaneDataManager

def draw_lane_line(img, points):
    print("points count:{}".format(len(points)))
    if len(points) > 0:
        for ps in points:
            ps = np.array(ps, dtype=np.int32)
            for i in range(len(ps)):
                if i+1 < len(ps):
                    p1 = ps[i]
                    p2 = ps[i+1]
                    print(p1, p2)
                    cv2.line(img, p1, p2, color=(0, 255, 0), thickness=2)
    return img

if __name__ == "__main__":
    lane_manage = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")

    print(lane_manage.get_data_len())
    print(lane_manage.get_lane_points(1))

    for i in range(lane_manage.get_data_len()):
        points, image_name = lane_manage.get_lane_points(i)
        image_file_path = os.path.join("/Users/take/fun/dataset/bdd100k/images/100k/train", image_name)
        img = cv2.imread(image_file_path)
        img = draw_lane_line(img, points)
        cv2.imshow("image", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
