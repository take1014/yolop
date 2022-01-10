#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import glob
import ast
import json
import cv2
import numpy as np

class BBoxDataManager(object):
    def __init__(self, json_path):
        self.json_path = json_path
        f = open(json_path)
        self.json_list = json.load(f)
        f.close()

    def get_data_len(self):
        return len(self.json_list)

    def get_bboxs(self, idx, category="car"):
        assert 0 <= idx < len(self.json_list)

        image_name = self.json_list[idx]["name"]

        bboxs = []
        if "labels" in self.json_list[idx]:
            for label in self.json_list[idx]["labels"]:
                if (label["category"] == category):
                    bbox = label["box2d"]
                    bboxs.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])

        return bboxs, image_name

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


def create_lane_json():

    if not os.path.exists("/Users/take/fun/dataset/bdd100k/json/lane"):
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/lane")
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/lane/train")
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/lane/val")

    lane_manage_train = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")
    for i in range(lane_manage_train.get_data_len()):
        points, name = lane_manage_train.get_points(i)
        file_name = "./lane/train/{}.json".format(name.replace(".jpg",""))
        dict_info = {"lane":points, "raw_file": name}
        print(file_name)
        with open(file_name, "w") as f:
            f.write(str(dict_info).replace("'",'"'))

    lane_manage_val = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_val.json")
    for i in range(lane_manage_val.get_data_len()):
        points, name = lane_manage_val.get_points(i)
        file_name = "./lane/val/{}.json".format(name.replace(".jpg",""))
        dict_info = {"lane":points, "raw_file": name}
        print(file_name)
        with open(file_name, "w") as f:
            f.write(str(dict_info).replace("'",'"'))

def create_bbox_json():

    if not os.path.exists("/Users/take/fun/dataset/bdd100k/json/bbox"):
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/bbox")
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/bbox/train")
        os.mkdir("/Users/take/fun/dataset/bdd100k/json/bbox/val")

    bbox_manage_train = BBoxDataManager("/Users/take/fun/dataset/bdd100k/labels/det_20/det_train.json")
    for i in range(bbox_manage_train.get_data_len()):
        bboxs, name = bbox_manage_train.get_bboxs(i)
        file_name = "./bbox/train/{}.json".format(name.replace(".jpg",""))
        dict_info = {"bbox":bboxs, "raw_file": name}
        print(file_name)
        with open(file_name, "w") as f:
            f.write(str(dict_info).replace("'",'"'))

    bbox_manage_val = BBoxDataManager("/Users/take/fun/dataset/bdd100k/labels/det_20/det_val.json")
    for i in range(bbox_manage_val.get_data_len()):
        bboxs, name = bbox_manage_val.get_bboxs(i)
        file_name = "./bbox/val/{}.json".format(name.replace(".jpg",""))
        dict_info = {"bbox":bboxs, "raw_file": name}
        print(file_name)
        with open(file_name, "w") as f:
            f.write(str(dict_info).replace("'",'"'))

if __name__ == "__main__":
    # create json file for lane, bounding box
    create_lane_json()
    create_bbox_json()

    # lane_list = glob.glob("./lane/train/*.json")
    # bbox_list = glob.glob("./bbox/train/*.json")
    # print(len(lane_list))
    # print(len(bbox_list))
    # for lane in lane_list:
    #     file_name = os.path.basename(lane)
    #     print(file_name.replace(".json",""))
    #     if not os.path.exists("./bbox/train/{}.json".format(file_name.replace(".json",""))):
    #         assert False

    # # validation
    # lane_list = glob.glob("./lane/val/*.json")
    # bbox_list = glob.glob("./bbox/val/*.json")
    # print(len(lane_list))
    # print(len(bbox_list))
    # for lane in lane_list:
    #     file_name = os.path.basename(lane)
    #     print(file_name.replace(".json",""))
    #     if not os.path.exists("./bbox/val/{}.json".format(file_name.replace(".json",""))):
    #         assert False

    # read json file
    # with open("./lane_train/lane_0000f77c-6257be58.json", "r") as f:
    #     for l in f:
    #         print(ast.literal_eval(l))
    #         print(type(ast.literal_eval(l)))
    #         print(l)
    #         print(type(l))

    # bbox_manage_train = BBoxDataManager("/Users/take/fun/dataset/bdd100k/labels/det_20/det_train.json")
    # print(bbox_manage_train.get_bboxs(5))
    # lane_manage_train = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")
    # lane_manage_val   = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_val.json")

    # # calc histogram
    # def calc_hist(manage, category_hist):
    #     for i in range(manage.get_data_len()):
    #         category = manage.get_category(i)
    #         for cat in category:
    #             category_hist[cat] += 1
    #     return category_hist

    # category_hist = { "road curb":0, "single white":0, "double white":0, "single yellow":0,
    #                   "double yellow":0, "single other":0, "double other":0, "crosswalk":0 }
    # # train data
    # print("train data count:{}".format(lane_manage_train.get_data_len()))
    # category_hist = calc_hist(lane_manage_train, category_hist)

    # # validation data
    # print("val data count:{}".format(lane_manage_val.get_data_len()))
    # category_hist = calc_hist(lane_manage_val, category_hist)

    # # show histogram
    # import matplotlib.pyplot as plt
    # x = [i for i in range(len(category_hist))]
    # plt.bar(x, category_hist.values(), tick_label=list(category_hist.keys()))
    # plt.show()
