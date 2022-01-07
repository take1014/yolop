#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import json
import cv2
import numpy as np

class LaneDataManager(object):
    def __init__(self, json_path):
        self.json_path = json_path
        f = open(json_path)
        self.json_list = json.load(f)
        f.close()

    def get_data_len(self):
        return len(self.json_list)

    def get_points(self, idx):

        assert 0 <= idx < len(self.json_list)

        image_name = self.json_list[idx]["name"]

        lane_types = ""
        points = []
        if "labels" in self.json_list[idx]:
            for label in self.json_list[idx]["labels"]:
                # if not (label["attributes"]["laneTypes"] == "crosswalk" and label["attributes"]["laneDirection"] == "vertical")
                if not (label["attributes"]["laneTypes"] == "crosswalk"):
                    for poly in label["poly2d"]:
                        line_dict = {"poly": poly["vertices"], "laneDir":label ["attributes"]["laneDirection"]}
                        points.append(line_dict)

        return points, image_name

    def get_category(self, idx):
        assert 0 <= idx < len(self.json_list)

        category = []
        if "labels" in self.json_list[idx]:
            for label in self.json_list[idx]["labels"]:
                category.append(label["category"])

        return category

if __name__ == "__main__":
    lane_manage_train = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")
    lane_manage_val   = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_val.json")

    # calc histogram
    def calc_hist(manage, category_hist):
        for i in range(manage.get_data_len()):
            category = manage.get_category(i)
            for cat in category:
                category_hist[cat] += 1
        return category_hist

    category_hist = { "road curb":0, "single white":0, "double white":0, "single yellow":0,
                      "double yellow":0, "single other":0, "double other":0, "crosswalk":0 }
    # train data
    print("train data count:{}".format(lane_manage_train.get_data_len()))
    category_hist = calc_hist(lane_manage_train, category_hist)

    # validation data
    print("val data count:{}".format(lane_manage_val.get_data_len()))
    category_hist = calc_hist(lane_manage_val, category_hist)

    # show histogram
    import matplotlib.pyplot as plt
    x = [i for i in range(len(category_hist))]
    plt.bar(x, category_hist.values(), tick_label=list(category_hist.keys()))
    plt.show()
