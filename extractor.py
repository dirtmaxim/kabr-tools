import os
import sys
from lxml import etree
import cv2
from src.utils import get_scene
from collections import OrderedDict
from src.detector import Detector
from src.tracker import Tracker, TrackedObject
from src.animal import Animal
from src.draw import Draw
from tqdm import tqdm


def extract(video_path, annotation_path, tracking):
    # Parse CVAT for video 1.1 annotation file.
    root = etree.parse(annotation_path).getroot()
    annotated = dict()

    for track in root.iterfind("track"):
        track_id = int(track.attrib["id"])

        for box in track.iter("box"):
            frame_id = int(box.attrib["frame"])

            if annotated.get(frame_id) is None:
                annotated[frame_id] = OrderedDict()

            annotated[frame_id][track_id] = [int(float(box.attrib["xtl"])),
                                             int(float(box.attrib["ytl"])),
                                             int(float(box.attrib["xbr"])),
                                             int(float(box.attrib["ybr"]))]

    if not os.path.exists(f"mini-scenes"):
        os.makedirs(f"mini-scenes")

    annotated_size = int("".join(root.find("meta").find("task").find("size").itertext()))
    name = os.path.splitext(video_path.split("/")[-1])[0]
    index = 0
    scene_width, scene_height = 400, 300
    vc = cv2.VideoCapture(video_path)
    original_width, original_height = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(f"mini-scenes/{name}.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"), 29.97,
                         (original_width, original_height))
    tracker = Tracker(max_disappeared=40, max_distance=200)
    tracked_objects = {}
    vc.set(cv2.CAP_PROP_POS_FRAMES, index)
    tracks_vw = dict()
    pbar = tqdm(total=annotated_size)

    while vc.isOpened():
        returned, frame = vc.read()

        if returned:
            visualization = frame.copy()

            if annotated.get(index) is not None:
                detections = []
                centroids = []
                objects = OrderedDict()
                colors = OrderedDict()

                for object_id, box in annotated[index].items():
                    detections.append(box)
                    centroid = Detector.get_centroid(box)
                    centroids.append(centroid)

                    if not tracking:
                        objects[object_id] = centroid
                        colors_values = list(tracker.colors_table.values())
                        colors[object_id] = colors_values[object_id % len(colors_values)]

                if tracking:
                    objects, colors = tracker.update(centroids)

                animals = Animal.animal_factory(objects, centroids, detections, colors)

                for animal in animals:
                    if tracks_vw.get(animal.object_id) is None:
                        if not os.path.exists(f"mini-scenes/{name}"):
                            os.makedirs(f"mini-scenes/{name}")

                        tracks_vw[animal.object_id] = cv2.VideoWriter(f"mini-scenes/{name}/{animal.object_id}.mp4",
                                                                      cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                                                      29.97, (scene_width, scene_height))
                for animal in animals:
                    TrackedObject.update(tracked_objects, animal)
                    TrackedObject.visualize_track(visualization, tracked_objects, animal, 20)
                    Draw.scene(visualization, animal, scene_width, scene_height)
                    Draw.animal_id(visualization, animal, scene=True)
                    scene_frame = frame.copy()
                    scene_frame = get_scene(scene_frame, animal, scene_width, scene_height)
                    tracks_vw[animal.object_id].write(scene_frame)

            cv2.putText(visualization, f"Frame: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.imshow("tracks2crops", cv2.resize(visualization, (original_width // 2, original_height // 2)))
            vw.write(visualization)
            key = cv2.waitKey(1)
            index += 1
            pbar.update(1)

            if key == 27:
                break
        else:
            break

    for track_key in tracks_vw.keys():
        tracks_vw[track_key].release()

    pbar.close()
    vc.release()
    vw.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("python extractor.py path_to_video path_to_annotation")
        exit(0)
    elif len(sys.argv) == 3:
        video = sys.argv[1]
        annotation = sys.argv[2]
        tracking = False
    # tracking=True: use external tracker instead of CVAT tracks.
    # tracking=False: use CVAT tracks.
    elif len(sys.argv) == 4:
        video = sys.argv[1]
        annotation = sys.argv[2]
        tracking = bool(sys.argv[3])

    if os.path.isdir(annotation):
        videos = []
        annotations = []

        for root, dirs, files in os.walk(annotation, topdown=False):
            for file in files:
                videos.append(os.path.join(video + root[len(annotation):], os.path.splitext(file)[0] + ".mp4"))
                annotations.append(os.path.join(root, file))

        for i, (video, annotation) in enumerate(zip(videos, annotations)):
            print(f"{i + 1}/{len(annotations)}:")

            if not os.path.exists(video):
                print(f"Path {video} does not exist.")
                continue

            extract(video, annotation, tracking)
    else:
        extract(video, annotation, tracking)
