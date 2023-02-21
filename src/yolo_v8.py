import numpy as np
from ultralytics import YOLO


class YOLOv8:
    def __init__(self, weights="yolov8x.pt", imgsz=640, conf=0.5):
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)
        print("Hello1")
        self.names = self.model.names

    def forward(self, image):
        width = image.shape[1]
        height = image.shape[0]
        boxes = self.model.predict(source=image[:, :, ::-1].astype(np.uint8), imgsz=self.imgsz, verbose=False)[
            0].boxes.cpu()
        bounding_boxes = boxes.xyxyn.numpy()
        classes = boxes.cls.numpy()
        confidences = boxes.conf.numpy()
        filtered = []

        for bounding_box, class_, confidence in zip(bounding_boxes, classes, confidences):
            if confidence > self.conf:
                if self.names[class_] in ["zebra", "horse", "giraffe"]:
                    bounding_box[0] = int(bounding_box[0] * width)
                    bounding_box[1] = int(bounding_box[1] * height)
                    bounding_box[2] = int(bounding_box[2] * width)
                    bounding_box[3] = int(bounding_box[3] * height)
                    confidence = float(f"{confidence:.2f}")
                    filtered.append(
                        [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3],
                         confidence, self.names[class_].capitalize()])

        return filtered

    @staticmethod
    def get_centroid(detection):
        x = int(detection[0] + (detection[2] - detection[0]) / 2)
        y = int(detection[1] + (detection[3] - detection[1]) / 2)

        return x, y
