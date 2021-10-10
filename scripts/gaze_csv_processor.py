# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import csv
import sys


class Point:
    def __init__(self, x_init: float, y_init: float, ts_init: float, vf_init: int = 0):
        self.x: float = x_init
        self.y: float = y_init
        self.ts: float = ts_init
        self.vf: int = vf_init

    def __lt__(self, other):
        if self.ts == other.ts:
            return self.ts < other.ts
        return self.ts < other.ts

    def __gt__(self, other):
        return other.__lt__(self)

    def __repr__(self) -> str:
        return "".join(["\nPoint(", str(self.x), ",", str(self.y), ") : time: ", str(self.ts)," frame: ", str(self.vf)])

class gaze_csv_processor:

    @staticmethod
    def get_nearest_frame(frames, time):
        lastdelta = 9999999
        currentanswer = -1
        for ts, frame in frames.items():
            delta = abs(time-ts)
            if delta < lastdelta:
                lastdelta = delta
                currentanswer = frame
            elif delta > lastdelta:
                break
        return currentanswer

    @staticmethod
    def process_gaze_data(path0: str, path1: str):
        # Open all the files required
        # get_fullpath_by_prefix('./002/surfaces/','srf_positons')
        try:
            with open(util.get_fullpath_by_prefix(path0, 'srf_positons'), 'r') as f:
                data_frames = list(csv.reader(f, delimiter=','))

            with open(util.get_fullpath_by_prefix(path0, 'gaze_positions_on_surface'), 'r') as f:
                data_base = list(csv.reader(f, delimiter=','))

            with open(util.get_fullpath_by_prefix(path1, 'gaze_positions_on_surface'), 'r') as f:
                data_second = list(csv.reader(f, delimiter=','))
        except IndexError as e:
            print("One of the exported files not found")
            sys.exit(1)

        except OSError as e:
            print("One of the exported files not found")
            sys.exit(1)

        # calculate (TS) - > (Frame Number) operator
        timestamp_to_frame = {}

        # Points dictionaties indexed by base video frames
        base_dict = {}
        second_dict = {}

        # Fill the timestamp_to_frame operator
        # Fill with empty lists

        for line in np.array(data_frames)[1:]:
            frame_id = int(line[0])
            timestamp_to_frame[float(line[1])] = frame_id
            base_dict[frame_id] = []  # point(-9999, -9999)
            second_dict[frame_id] = []

        for line in data_second[1:]:
            base_frame_number = gaze_csv_processor.get_nearest_frame(timestamp_to_frame, float(line[2]))
            second_dict[base_frame_number].append(Point(float(line[3]), float(line[4]), float(line[2]), base_frame_number))

        for line in np.array(data_base)[1:]:
            base_dict[int(line[1])].append(Point(float(line[3]), float(line[4]), float(line[2]), int(line[1])))

        return base_dict, second_dict


class util:

    @staticmethod
    def get_fullpath_by_prefix(path_to_search, prefix):
        try:
            path = os.path.join(path_to_search, prefix)
            print (path, glob.glob(path))
            return glob.glob(os.path.join(path_to_search, prefix) + "*")[0]
        except Exception as e:
            print("Surfaces not found\n", e)
            sys.exit(1)