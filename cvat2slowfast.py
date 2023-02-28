import os
import sys
import json
from lxml import etree
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import cv2

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python cvat2slowfast.py path_to_mini_scenes dataset_name [zebra, giraffe]")
        exit(0)
    elif len(sys.argv) == 4:
        path_to_mini_scenes = sys.argv[1]
        dataset_name = sys.argv[2]
        subject = sys.argv[3]

    label2number_zebra = {"Walk": 0,
                          "Graze": 1,
                          "Head Up": 2,
                          "Auto-Groom": 3,
                          "Mutual Grooming": 4,
                          "Trotting": 5,
                          "Running": 6,
                          "Drinking": 7,
                          "Herding": 8,
                          "Lying Down": 9,
                          "Mounting-Mating": 10,
                          "Sniff": 11,
                          "Urinating": 12,
                          "Defecating": 13,
                          "Dusting": 14,
                          "Fighting": 15,
                          "Chasing": 16,
                          "Occluded": 17,
                          "Out of Focus": 18,
                          "Out of Frame": 19}

    number2label_zebra = {value: key for key, value in label2number_zebra.items()}

    label2number_giraffe = {"Walking": 0,
                            "Head Up": 1,
                            "Auto-Groom": 2,
                            "Mutual Grooming": 3,
                            "Running": 4,
                            "Browsing": 5,
                            "Mounting": 6,
                            "Urinating": 7,
                            "Defecating": 8,
                            "Fighting": 9,
                            "Occluded": 10,
                            "Out of Focus": 11,
                            "Out of Frame": 12}

    number2label_giraffe = {value: key for key, value in label2number_giraffe.items()}

    if subject == "zebra":
        label2number = label2number_zebra
        number2label = number2label_zebra
    elif subject == "giraffe":
        label2number = label2number_giraffe
        number2label = number2label_giraffe
    else:
        raise NotImplemented(f"{subject.capitalize()} is not supported.")

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    if not os.path.exists(f"{dataset_name}/annotation"):
        os.makedirs(f"{dataset_name}/annotation")

    if not os.path.exists(f"{dataset_name}/dataset/image"):
        os.makedirs(f"{dataset_name}/dataset/image")

    with open(f"{dataset_name}/annotation/classes.json", "w") as file:
        json.dump(label2number, file)

    headers = {"original_vido_id": [], "video_id": pd.Series(dtype="int"), "frame_id": pd.Series(dtype="int"),
               "path": [], "labels": []}
    charades_df = pd.DataFrame(data=headers)
    video_id = 1

    for folder in os.listdir(path_to_mini_scenes):
        if os.path.exists(f"{path_to_mini_scenes}/{folder}/actions"):
            for file in os.listdir(f"{path_to_mini_scenes}/{folder}/actions"):
                if os.path.splitext(file)[1] == ".xml":
                    annotation_file = f"{path_to_mini_scenes}/{folder}/actions/{file}"
                    video_file = f"{path_to_mini_scenes}/{folder}/{os.path.splitext(file)[0]}.mp4"
                    output_folder = f"{dataset_name}/dataset/image/{folder}_{os.path.splitext(file)[0]}"

                    if not os.path.exists(video_file):
                        print(f"{video_file} does not exist.")
                        continue

                    print(f"{annotation_file} | {video_file} -> {output_folder}")
                    sys.stdout.flush()

                    root = etree.parse(annotation_file).getroot()
                    annotated = OrderedDict()

                    for track in root.iterfind("track"):
                        for entry in track.iter("points"):
                            frame_id = entry.attrib["frame"]
                            outside = entry.attrib["outside"]

                            if outside == "1":
                                continue

                            behavior = "".join(entry.find("attribute").itertext())

                            if annotated.get(frame_id) is None:
                                annotated[frame_id] = OrderedDict()

                            annotated[frame_id] = behavior

                    index = 0
                    vc = cv2.VideoCapture(video_file)
                    size = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
                    pbar = tqdm(total=size)

                    while vc.isOpened():
                        returned, frame = vc.read()

                        if returned:
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

                            cv2.imwrite(f"{output_folder}/{index}.jpg", frame)
                            charades_df.loc[len(charades_df.index)] = [f"{folder}_{os.path.splitext(file)[0]}",
                                                                       video_id,
                                                                       index,
                                                                       f"{folder}_{os.path.splitext(file)[0]}/{index}.jpg",
                                                                       str(label2number[annotated[str(index)]])]
                            index += 1
                            pbar.update(1)
                        else:
                            break

                    pbar.close()
                    vc.release()
                    video_id += 1

    size = charades_df.shape[0]
    split = 0.7
    train_size = int(size * split)

    # Find position of a next video to prevent data leak.
    current_video_id = charades_df.iloc[[train_size]]["video_id"].item()

    for i in range(train_size, size):
        if charades_df.iloc[[i]]["video_id"].item() != current_video_id:
            train_size = i
            break

    train = charades_df.iloc[:train_size]
    val = charades_df.iloc[train_size:]

    train.to_csv(f"{dataset_name}/annotation/train.csv", sep=" ", index=False)
    val.to_csv(f"{dataset_name}/annotation/val.csv", sep=" ", index=False)
