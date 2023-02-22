import numpy as np
import cv2


class Draw:
    @staticmethod
    def bounding_box(image, animal, radius=20, length=15, thickness=4):
        if animal.detection is not None:
            x_start, y_start = animal.detection[0], animal.detection[1]
            x_end, y_end = animal.detection[2], animal.detection[3]
            cv2.line(image, (x_start + radius, y_start), (x_start + radius + length, y_start), animal.color, thickness)
            cv2.line(image, (x_start, y_start + radius), (x_start, y_start + radius + length), animal.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_start + radius), (radius, radius), 180, 0, 90, animal.color,
                        thickness)
            cv2.line(image, (x_end - radius, y_start), (x_end - radius - length, y_start), animal.color, thickness)
            cv2.line(image, (x_end, y_start + radius), (x_end, y_start + radius + length), animal.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_start + radius), (radius, radius), 270, 0, 90, animal.color,
                        thickness)
            cv2.line(image, (x_start + radius, y_end), (x_start + radius + length, y_end), animal.color, thickness)
            cv2.line(image, (x_start, y_end - radius), (x_start, y_end - radius - length), animal.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_end - radius), (radius, radius), 90, 0, 90, animal.color, thickness)
            cv2.line(image, (x_end - radius, y_end), (x_end - radius - length, y_end), animal.color, thickness)
            cv2.line(image, (x_end, y_end - radius), (x_end, y_end - radius - length), animal.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_end - radius), (radius, radius), 0, 0, 90, animal.color, thickness)

    @staticmethod
    def scene(image, animal, scene_width=400, scene_height=300, thickness=4):

        if animal.centroid is not None:
            start = (
                np.max([0, animal.centroid[0] - scene_width // 2]), np.max([0, animal.centroid[1] - scene_height // 2]))
            end = (np.min([image.shape[1] - 1, animal.centroid[0] + scene_width // 2]),
                   np.min([image.shape[0] - 1, animal.centroid[1] + scene_height // 2]))
            cv2.rectangle(image, start, end, animal.color, thickness)

    @staticmethod
    def actions(image, animal, actions, margin=150):
        thickness_out = 25
        thickness_in = 2
        size = 0.6
        label = " | ".join(actions)
        border_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_out)
        label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)
        centroid = animal.centroid

        cv2.putText(image, label,
                    (np.max([0, centroid[0] - border_length[0][0] // 2 + 10]),
                     np.max([0, centroid[1] + margin])),
                    cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i + 100 for i in animal.color]), thickness_out, cv2.LINE_AA)
        cv2.putText(image, label,
                    (np.max([0, centroid[0] - label_length[0][0] // 2]),
                     np.max([0, centroid[1] + margin])),
                    cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 100 for i in animal.color]), thickness_in, cv2.LINE_AA)

    @staticmethod
    def animal_id(image, animal, scene=False):
        if animal.class_ is not None and animal.confidence is not None:
            label = f"#{animal.object_id}: {animal.confidence} {animal.class_}"
        else:
            label = f"#{animal.object_id}"

        border_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25)
        label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        if not scene:
            cv2.putText(image, label,
                        (np.max([0, animal.centroid[0] - border_length[0][0] // 2 + 10]),
                         np.max([0, animal.centroid[1] - 30])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i + 30 for i in animal.color]), 25, cv2.LINE_AA)
            cv2.putText(image, label,
                        (np.max([0, animal.centroid[0] - label_length[0][0] // 2]),
                         np.max([0, animal.centroid[1] - 30])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i - 30 for i in animal.color]), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, label,
                        (np.max([0, animal.centroid[0] - border_length[0][0] // 2 + 10]),
                         np.max([0, animal.centroid[1] - 150])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i + 30 for i in animal.color]), 25, cv2.LINE_AA)
            cv2.putText(image, label,
                        (np.max([0, animal.centroid[0] - label_length[0][0] // 2]),
                         np.max([0, animal.centroid[1] - 150])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i - 30 for i in animal.color]), 2, cv2.LINE_AA)
