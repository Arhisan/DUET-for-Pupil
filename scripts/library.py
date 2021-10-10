from bisect import bisect_left
from typing import List

from gaze_csv_processor import Point


def get_nearest_point(points: List[Point], original_point: Point) -> Point:
    last_delta = 9999999
    current_answer = None
    # for point in points[original_point.vf]:
    #     delta = abs(original_point.ts-point.ts)
    #     if delta < last_delta:
    #         last_delta = delta
    #         current_answer = point
    #     elif delta > last_delta:
    #         break
    # return current_answer
    i = bisect_left(points, original_point)

    if i == 0:
        return points[0]

    if i == len(points):
        return points[len(points) - 1]

    left = points[i - 1]
    right = points[i]

    if original_point.ts - left.ts < right.ts - original_point.ts:
        return left
    return right
    #
    # print("INDEX:", i)
    #
    # for point_list in points.values():
    #     for point in point_list:
    #         delta = abs(original_point.ts - point.ts)
    #         if delta < last_delta:
    #             last_delta = delta
    #             current_answer = point
    #         elif delta > last_delta:
    #             break
    # return current_answer