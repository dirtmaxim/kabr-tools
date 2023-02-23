import numpy as np
from ultralytics import YOLO
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction


class YOLOv8:
    def __init__(self, weights="yolov8x.pt", imgsz=640, conf=0.5):
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)
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

                    if self.names[class_] == "horse":
                        label = "Zebra"
                    else:
                        label = self.names[class_].capitalize()

                    filtered.append(
                        [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3],
                         confidence, label])

        return filtered

    @staticmethod
    def get_centroid(detection):
        x = int(detection[0] + (detection[2] - detection[0]) / 2)
        y = int(detection[1] + (detection[3] - detection[1]) / 2)

        return x, y


class YOLOv8_SAHI(YOLOv8):
    def __init__(self, weights="yolov8x.pt", slice_height=640, slice_width=640,
                 overlap_height_ratio=0.2, overlap_width_ratio=0.2, conf=0.5):
        super().__init__(weights, imgsz=None, conf=conf)
        self.model = Yolov8DetectionModel(model=self.model)
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def forward(self, image):
        boxes = get_sliced_prediction(image[:, :, ::-1].astype(np.uint8), self.model, slice_height=self.slice_height,
                                      slice_width=self.slice_width, overlap_height_ratio=self.overlap_height_ratio,
                                      overlap_width_ratio=self.overlap_width_ratio, verbose=False).to_coco_predictions()
        filtered = []

        for box in boxes:
            if box["score"] > self.conf:
                if box["category_name"] in ["zebra", "horse", "giraffe"]:
                    x_start = int(box["bbox"][0])
                    y_start = int(box["bbox"][1])
                    x_end = int(box["bbox"][0] + box["bbox"][2])
                    y_end = int(box["bbox"][1] + box["bbox"][3])
                    confidence = float(f"{box['score']:.2f}")

                    if box["category_name"] == "horse":
                        label = "Zebra"
                    else:
                        label = box["category_name"].capitalize()

                    filtered.append([x_start, y_start, x_end, y_end, confidence, label])

        return filtered
