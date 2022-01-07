import os
import json
import cv2
import numpy as np

class LaneDataManager(object):
    def __init__(self, json_path):
        self.json_path = json_path
        f = open("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")
        self.json_file = json.load(f)
        f.close()

    def get_data_len(self):
        return len(self.json_file)

    def get_lane_points(self, idx):

        assert 0 <= idx < len(self.json_file)

        image_name = self.json_file[idx]["name"]

        lane_types = ""
        points = []
        if "labels" in self.json_file[idx]:
            for label in self.json_file[idx]["labels"]:
                if not (label["attributes"]["laneTypes"] == "crosswalk" and label["attributes"]["laneDirection"] == "vertical"):
                    for poly in label["poly2d"]:
                        points.append(poly["vertices"])

        return points, image_name

if __name__ == "__main__":
    lane_manage = LaneDataManager("/Users/take/fun/dataset/bdd100k/labels/lane/polygons/lane_train.json")
    print(lane_manage.get_data_len())
    print(lane_manage.get_lane_points(30))
