import numpy as np


class Animal:
    def __init__(self, object_id=None, centroid=None, detection=None, color=None):

        self.object_id = object_id
        self.centroid = centroid
        self.color = color

        if detection is not None:
            self.detection = (int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]))
        else:
            self.detection = None

    @staticmethod
    def animal_factory(objects, centroids, detections, colors):
        animals = []

        for object_id, centroid in objects.items():
            assigned = False

            for i, c in enumerate(centroids):
                if np.array_equal(centroid, c):
                    animals.append(Animal(object_id, centroid, detections[i], colors[object_id]))
                    assigned = True

            if not assigned:
                # Object disappeared for some frames.
                animals.append(Animal(object_id, centroid, None, colors[object_id]))

        return animals
