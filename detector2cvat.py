import os
import sys
import cv2
import numpy as np
import datetime
from lxml import etree
from collections import OrderedDict
from tqdm import tqdm
import shutil
from src.yolo_v8 import YOLOv8
from src.detector import Detector
from src.tracker import Tracker, TrackedObject
from src.animal import Animal
from src.draw import Draw

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python detector2cvat path_to_videos path_to_save")
        exit(0)
    else:
        path_to_videos = sys.argv[1]
        path_to_save = sys.argv[2]

    videos = []

    for root, dirs, files in os.walk(path_to_videos):
        for file in files:
            if os.path.splitext(file)[1] == ".mp4":
                videos.append(f"{root}/{file}")

    model = YOLOv8("yolov8x.pt", imgsz=1536, conf=0.5)

    for i, video in enumerate(videos):
        name = os.path.splitext(video.split("/")[-1])[0]

        output_folder = path_to_save + os.sep + "/".join(os.path.splitext(video)[0].split("/")[-3:-1])
        output_path = f"{output_folder}/{name}.xml"
        print(f"{i + 1}/{len(videos)}: {video} -> {output_path}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        vc = cv2.VideoCapture(video)
        size = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(f"{output_folder}/{name}_demo.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                             29.97, (width, height))
        max_disappeared = 40
        tracker = Tracker(max_disappeared=max_disappeared, max_distance=300)
        tracked_objects = {}
        tracks_history = {}

        # Create CVAT for video 1.1 XML.
        xml_page = etree.Element("annotations")
        etree.SubElement(xml_page, "version").text = "1.1"
        xml_meta = etree.SubElement(xml_page, "meta")
        xml_task = etree.SubElement(xml_meta, "task")
        etree.SubElement(xml_task, "size").text = str(size)
        xml_original_size = etree.SubElement(xml_task, "original_size")
        etree.SubElement(xml_original_size, "width").text = str(width)
        etree.SubElement(xml_original_size, "height").text = str(height)
        etree.SubElement(xml_task, "source").text = f"{name}"

        index = 0
        vc.set(cv2.CAP_PROP_POS_FRAMES, index)
        annotated = OrderedDict()
        pbar = tqdm(total=size)

        while vc.isOpened():
            returned, frame = vc.read()

            if returned:
                visualization = frame.copy()
                detections = model.forward(frame)
                centroids = []

                for detection in detections:
                    centroids.append(YOLOv8.get_centroid(detection))

                objects, colors = tracker.update(centroids)
                animals = Animal.animal_factory(objects, centroids, detections, colors)

                for animal in animals:
                    TrackedObject.update(tracked_objects, animal)
                    TrackedObject.visualize_track(visualization, tracked_objects, animal, 20)
                    Draw.bounding_box(visualization, animal)
                    Draw.animal_id(visualization, animal)

                    if animal.class_ is not None:
                        if tracks_history.get(animal.object_id) is None:
                            tracks_history[animal.object_id] = {}

                        tracks_history[animal.object_id]["class_"] = animal.class_

                    if animal.detection is not None:
                        if tracks_history.get(animal.object_id) is None:
                            tracks_history[animal.object_id] = {}

                        tracks_history[animal.object_id]["detection"] = animal.detection

                for animal in animals:
                    if animal.class_ is not None:
                        class_ = animal.class_
                    else:
                        class_ = tracks_history[animal.object_id]["class_"]

                    if animal.detection is not None:
                        detection = animal.detection
                    else:
                        detection = tracks_history[animal.object_id]["detection"]

                    if annotated.get((str(animal.object_id), class_)) is None:
                        annotated[(str(animal.object_id), class_)] = []

                    annotated[(str(animal.object_id), class_)].append({"frame": str(index),
                                                                       "xtl": str(float(detection[0])),
                                                                       "ytl": str(float(detection[1])),
                                                                       "xbr": str(float(detection[2])),
                                                                       "ybr": str(float(detection[3]))})

                cv2.putText(visualization, f"Frame: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow("detector2cvat", cv2.resize(visualization, (int(width // 2.5), int(height // 2.5))))
                vw.write(visualization)
                key = cv2.waitKey(1)
                index += 1
                pbar.update(1)

                if key == 27:
                    break
            else:
                break

        pbar.close()
        vc.release()
        vw.release()
        cv2.destroyAllWindows()

        # Save annotations.
        for (track_id, label), boxes in annotated.items():
            # Remove short tracks.
            if len(tracked_objects[int(track_id)].centroids) > max_disappeared * 2 + 1:
                xml_track = etree.Element("track", id=track_id, label=label, source="manual")
                # Compensate tracker's last boxes.
                boxes = boxes[:-max_disappeared]

                for j, box in enumerate(boxes):
                    # Mark the end of the track.
                    if j < len(boxes) - 1:
                        outside = "0"
                    else:
                        outside = "1"

                    xml_box = etree.Element("box", frame=box["frame"], outside=outside, occluded="0", keyframe="1",
                                            xtl=box["xtl"], ytl=box["ytl"], xbr=box["xbr"], ybr=box["ybr"], z_order="0")
                    xml_track.append(xml_box)

                xml_page.append(xml_track)
            else:
                print(f"Auto-removed track #{track_id}.")

        xml_document = etree.ElementTree(xml_page)
        xml_document.write(output_path, xml_declaration=True, pretty_print=True, encoding="utf-8")
