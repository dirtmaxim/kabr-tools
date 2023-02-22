import os
import sys
import json
from lxml import etree
from collections import OrderedDict


def extract(annotation_path):
    # Parse CVAT for video 1.1 annotation file.
    root = etree.parse(annotation_path).getroot()
    annotated = dict()

    for track in root.iterfind("track"):
        for entry in track.iter("points"):
            frame_id = int(entry.attrib["frame"])
            behavior = "".join(entry.find("attribute").itertext())

            if annotated.get(frame_id) is None:
                annotated[frame_id] = OrderedDict()

            annotated[frame_id] = behavior

    video_path_id = "|".join(annotation_path.split("/")[-4:-1])
    track_path_id = os.path.splitext(annotation_path.split("/")[-1])[0]

    if not os.path.exists(f"mini-scenes/{video_path_id}/actions"):
        os.makedirs(f"mini-scenes/{video_path_id}/actions")

    with open(f"mini-scenes/{video_path_id}/actions/{track_path_id}.json", "w") as file:
        print(f"Saved to: mini-scenes/{video_path_id}/actions/{track_path_id}.json")
        json.dump(annotated, file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python actions_extractor.py path_to_annotations")
        exit(0)
    elif len(sys.argv) == 2:
        annotation = sys.argv[1]

    if os.path.isdir(annotation):
        annotations = []

        for root, dirs, files in os.walk(annotation):
            for file in files:
                annotations.append(os.path.join(root, file))

        for i, annotation in enumerate(annotations):
            print(f"{i + 1}/{len(annotations)}: {annotation}")
            extract(annotation)
    else:
        extract(annotation)
