from gaze_csv_processor import Point
from library import get_nearest_point


def test_get_nearest_point():
    points = [Point(1, 0, 1, 0), Point(2, 0, 5, 0), Point(3, 0, 2, 0), Point(4, 0, 6, 0)]
    points.sort(key=lambda x: x.ts)

    result = get_nearest_point(points, Point(1, 1, 3, 0))

    assert result.ts == 2
