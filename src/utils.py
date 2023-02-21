import numpy as np
import cv2


def get_scene(image, animal, scene_width, scene_height):
    width = scene_width // 2
    height = scene_height // 2

    if animal.centroid is not None:
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0

        start_x = np.max([0, animal.centroid[0] - width])
        start_y = np.max([0, animal.centroid[1] - height])
        end_x = np.min([image.shape[1] - 1, animal.centroid[0] + width])
        end_y = np.min([image.shape[0] - 1, animal.centroid[1] + height])

        crop = image[start_y:end_y, start_x:end_x]

        if animal.centroid[0] - width < 0:
            pad_left = width - animal.centroid[0]

        if animal.centroid[1] - height < 0:
            pad_top = height - animal.centroid[1]

        if animal.centroid[0] + width > image.shape[1] - 1:
            pad_right = animal.centroid[0] + width - image.shape[1] + 1

        if animal.centroid[1] + height > image.shape[0] - 1:
            pad_bottom = animal.centroid[1] + height - image.shape[0] + 1

        padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        return padded
    else:
        return image
