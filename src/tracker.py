import numpy as np
from collections import OrderedDict
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import cv2


class Tracker:
    def __init__(self, max_disappeared, max_distance):
        """Track objects. mode is parameter which defines algorithm for data association.
        max_disappeared is maximum amount of frames allowed to not detect object.
        max distance is maximum leap for object between frames."""

        self.next_object_id = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.kalman_filters = OrderedDict()
        self.colors = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.colors_table = OrderedDict([
            ("Nebulas Blue", (170, 105, 63)),
            ("Valiant Poppy", (58, 61, 189)),
            ("Ultra Violet", (149, 91, 107)),
            ("Red Pear", (69, 65, 127)),
            ("Ceylon Yellow", (65, 174, 213)),
            ("Martini Olive", (87, 111, 118)),
            ("Russet Orange", (46, 122, 228)),
            ("Crocus Petal", (201, 158, 190)),
            ("Limelight", (127, 234, 241)),
            ("Quetzal Green", (109, 110, 0)),
            ("Niagara", (169, 140, 87)),
            ("Primrose Yellow", (85, 209, 246)),
            ("Lapis Blue", (141, 75, 0)),
            ("Flame", (44, 85, 242)),
            ("Island Paradise", (227, 222, 149)),
            ("Pale Dogwood", (194, 205, 237)),
            ("Pink Yarrow", (117, 49, 206)),
            ("Kale", (71, 114, 90)),
            ("Hazelnut", (149, 176, 207))
        ])

    def add(self, centroid):
        """Add new object."""

        self.objects[self.next_object_id] = np.array(centroid)
        self.disappeared[self.next_object_id] = 0
        self.kalman_filters[self.next_object_id] = KalmanFiler()
        colors_values = list(self.colors_table.values())
        self.colors[self.next_object_id] = colors_values[self.next_object_id % len(colors_values)]
        self.next_object_id += 1

    def delete(self, object_id):
        """Delete disappeared object."""

        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.kalman_filters[object_id]

    def update(self, centroids):
        """Update tracks on a new frame."""

        centroids = np.array(centroids)

        for object_id in self.objects.keys():
            self.objects[object_id] = self.kalman_filters[object_id].predict(self.objects[object_id][0],
                                                                             self.objects[object_id][1])

        if len(centroids) == 0:
            keys = list(self.disappeared.keys())

            for object_id in keys:
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.delete(object_id)

            return self.objects, self.colors

        if len(self.objects) == 0:
            for i in range(0, len(centroids)):
                self.add(centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            Y = distance.cdist(np.array(object_centroids), centroids)
            row_ind, col_ind = linear_sum_assignment(Y)
            used_rows = set()
            used_cols = set()

            for row, col in zip(row_ind, col_ind):
                if row in used_rows or col in used_cols:
                    continue

                if Y[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, Y.shape[0])).difference(used_rows)
            unused_cols = set(range(0, Y.shape[1])).difference(used_cols)

            if Y.shape[0] >= Y.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.delete(object_id)
            else:
                for col in unused_cols:
                    self.add(centroids[col])

        return self.objects, self.colors

    def reset(self):
        """Reset tracker to default state."""

        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.colors = OrderedDict()


class KalmanFiler:
    def __init__(self):
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                         [0, 1, 0, 0]], dtype=np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 0.01, 0],
                                                        [0, 1, 0, 0.01],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], dtype=np.float32)
        self.initial_x = None
        self.initial_y = None

    def predict(self, x, y):
        if self.initial_x is None:
            self.initial_x = x
            self.initial_y = y

            return np.array([x, y])

        self.kalman_filter.correct(np.array([x - self.initial_x, y - self.initial_y], dtype=np.float32))
        predicted = self.kalman_filter.predict()
        predicted_x = int(predicted[0][0]) + self.initial_x
        predicted_y = int(predicted[1][0]) + self.initial_y

        return np.array([predicted_x, predicted_y])


class Track:
    def __init__(self, object_id, color, label=None):
        self.object_id = object_id
        self.color = color
        self.label = label
        self.centroids = []
        self.boxes = []
        self.indices = []
        self.interpolated = []
        self.index2centroid = {}
        self.index2box = {}
        self.missed_boxes = 0


class Tracks:
    def __init__(self, max_disappeared, interpolation=True):
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.interpolation = interpolation
        self.refined_index = -1
        self.refined_tracks = []

    def __getitem__(self, key):
        return self.tracks[key]

    def __setitem__(self, key, value):
        if isinstance(value, Track):
            self.tracks[key] = value
        else:
            raise ValueError(f"Type {type(value)} is not supported.")

    def keys(self):
        return self.tracks.keys()

    def values(self):
        return self.tracks.values()

    def items(self):
        return self.tracks.items()

    def update(self, index, object):
        if self.tracks.get(object.object_id) is None:
            if object.label is None:
                self.tracks[object.object_id] = Track(object.object_id, object.color)
            else:
                self.tracks[object.object_id] = Track(object.object_id, object.color, object.label)

        self.tracks[object.object_id].centroids.append(object.centroid)
        self.tracks[object.object_id].index2centroid[index] = object.centroid

        if object.box is None:
            self.tracks[object.object_id].boxes.append(None)
            self.tracks[object.object_id].index2box[index] = None
        else:
            self.tracks[object.object_id].boxes.append(object.box)
            self.tracks[object.object_id].index2box[index] = object.box

        self.tracks[object.object_id].indices.append(index)
        self.tracks[object.object_id].interpolated.append(False)

        if self.interpolation:
            if self.tracks[object.object_id].boxes[-1] is None:
                self.tracks[object.object_id].missed_boxes += 1
            elif self.tracks[object.object_id].missed_boxes > 0:
                current_box = self.tracks[object.object_id].boxes[-1]
                past_box = self.tracks[object.object_id].boxes[-self.tracks[object.object_id].missed_boxes - 2]

                coordinates_0 = np.linspace(current_box[0], past_box[0],
                                            self.tracks[object.object_id].missed_boxes + 2, dtype=np.int32)[1:-1]
                coordinates_1 = np.linspace(current_box[1], past_box[1],
                                            self.tracks[object.object_id].missed_boxes + 2, dtype=np.int32)[1:-1]
                coordinates_2 = np.linspace(current_box[2], past_box[2],
                                            self.tracks[object.object_id].missed_boxes + 2, dtype=np.int32)[1:-1]
                coordinates_3 = np.linspace(current_box[3], past_box[3],
                                            self.tracks[object.object_id].missed_boxes + 2, dtype=np.int32)[1:-1]
                interpolated_boxes = np.vstack((coordinates_0, coordinates_1, coordinates_2, coordinates_3)).T

                for i, box in enumerate(interpolated_boxes):
                    box = list(box)
                    current = len(self.tracks[object.object_id].boxes) - 2 - i
                    self.tracks[object.object_id].boxes[current] = box
                    self.tracks[object.object_id].index2box[current] = box
                    self.tracks[object.object_id].interpolated[current] = True

                self.tracks[object.object_id].missed_boxes = 0

        if index != self.refined_index:
            self.refined_index = index

            for object_id in list(self.tracks.keys()):
                if object_id in self.refined_tracks:
                    continue

                last = self.tracks[object_id].indices[-1]

                if index - last > self.max_disappeared:
                    if len(self.tracks[object_id].indices) > self.max_disappeared * 2:
                        # Refined track.
                        self.tracks[object_id].centroids = self.tracks[object_id].centroids[:-self.max_disappeared]
                        self.tracks[object_id].boxes = self.tracks[object_id].boxes[:-self.max_disappeared]
                        self.tracks[object_id].indices = self.tracks[object_id].indices[:-self.max_disappeared]
                        self.refined_tracks.append(object_id)
                    else:
                        # Auto-removed track.
                        del self.tracks[object_id]
