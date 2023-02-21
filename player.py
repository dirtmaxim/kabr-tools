import os
import sys
import json
import numpy as np
from collections import OrderedDict
import cv2


def on_slider_change(value):
    global index, vcs, current, trackbar_position, paused, updated
    index = value

    if abs(trackbar_position - index) > 10:
        vcs[current].set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])

        if paused:
            updated = True


def pad(image, width, height):
    shape_0, shape_1 = image.shape[0], image.shape[1]

    if shape_0 < shape_1:
        new_width = int((height / shape_0) * shape_1)
        pad_size = (width - new_width) // 2
        image = cv2.resize(image, (new_width, height))
        padded = cv2.copyMakeBorder(image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        new_height = int((width / shape_1) * shape_0)
        pad_size = (height - new_height) // 2
        image = cv2.resize(image, (width, new_height))
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded


def draw_aim(current, image):
    if current == "main":
        return image

    copied = image.copy()
    x = image.shape[1] // 2
    y = image.shape[0] // 2
    cv2.line(image, (x - 20, y), (x + 20, y), (127, 127, 127), 2)
    cv2.line(image, (x, y - 20), (x, y + 20), (127, 127, 127), 2)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_id(current, image, metadata, width):
    if current == "main":
        label = f"Drone View"
        color = (127, 127, 127)
    else:
        label = f"Track #{current}"
        color = metadata["colors"][current]

    thickness_in = 4
    size = 1.5
    label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)

    copied = image.copy()
    cv2.rectangle(image, (width // 2 - label_length[0][0] // 2 - 50, 95),
                  (width // 2 + label_length[0][0] // 2 + 50, 180), (255, 255, 255), -1)
    cv2.putText(image, label, ((width - label_length[0][0]) // 2, 150),
                cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 50 for i in color]), thickness_in, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_actions(current, index, image, actions, metadata, width, height):
    if current == "main":
        return image

    if actions.get(current) is None:
        return image
    elif actions[current].get(str(index)) is None:
        return image

    color = metadata["colors"][current]
    label = "|".join(actions[current][str(index)].split(","))
    thickness_in = 4
    size = 1.5
    label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)

    copied = image.copy()
    cv2.rectangle(image, (width // 2 - label_length[0][0] // 2 - 50, height - 95),
                  (width // 2 + label_length[0][0] // 2 + 50, height - 180), (255, 255, 255), -1)
    cv2.putText(image, label, ((width - label_length[0][0]) // 2, height - 130),
                cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 50 for i in color]), thickness_in, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_info(image, width):
    copied = image.copy()
    cv2.rectangle(image, (width - 600, 100), (width - 100, 340), (0, 0, 0), -1)
    cv2.putText(image, "[0-9]: Show Track #[0-9]", (width - 565, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Shift + [0-9]: Show Track #[10-19]", (width - 565, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Enter: Show Drone View", (width - 565, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Esc: Exit", (width - 565, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("python player.py path_to_folder [save]")
        exit(0)
    elif len(sys.argv) == 2:
        folder = sys.argv[1]
        save = False
    elif len(sys.argv) == 3:
        folder = sys.argv[1]
        save = bool(sys.argv[2])

    name = folder.split("/")[-1]

    metadata_path = f"{folder}/metadata/{name}.json"
    actions_path = f"{folder}/actions"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    vcs = OrderedDict()
    vcs["main"] = cv2.VideoCapture(f"{folder}/{name}.mp4")

    for file in os.listdir(folder):
        if os.path.splitext(file)[-1] == ".mp4":
            if not os.path.splitext(file)[0].startswith(name):
                vcs[os.path.splitext(file)[0]] = cv2.VideoCapture(f"{folder}/{file}")

    actions = OrderedDict()

    if os.path.exists(actions_path):
        for file in os.listdir(actions_path):
            if os.path.splitext(file)[-1] == ".json":
                with open(f"{actions_path}/{file}", "r") as json_file:
                    actions[os.path.splitext(file)[0]] = json.load(json_file)

    index = 0
    cv2.namedWindow("TrackPlayer")
    cv2.createTrackbar(name, "TrackPlayer", index, len(metadata["main"]) - 1, on_slider_change)
    current = "main"
    vc = vcs[current]
    target_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    letter2hotkey = {41: "10", 33: "11", 64: "12", 35: "13", 36: "14", 37: "15", 94: "16", 38: "17", 42: "18", 40: "19"}
    paused, updated = False, False

    if save:
        vw = cv2.VideoWriter(f"{folder}/{name}_demo.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                             29.97, (target_width, target_height))

    while vc.isOpened():
        if index < len(metadata[current]):
            if metadata[current][index] < 0:
                current = "main"
                vc = vcs[current]
                vc.set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])

        returned, frame = vc.read()

        if returned:
            updated = False
            visualization = frame.copy()

            if current != "main":
                visualization = pad(visualization, target_width, target_height)

            visualization = draw_aim(current, visualization)
            visualization = draw_id(current, visualization, metadata, target_width)
            visualization = draw_actions(current, index, visualization, actions, metadata, target_width, target_height)
            visualization = draw_info(visualization, target_width)
            trackbar_position = cv2.getTrackbarPos(name, "TrackPlayer")
            cv2.setTrackbarPos(name, "TrackPlayer", index)

            if current != "main":
                cv2.putText(visualization, f"Frame: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow("TrackPlayer", cv2.resize(visualization, (int(target_width // 2.5), int(target_height // 2.5))))

            if save:
                vw.write(visualization)

            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key == 32 or paused:
                paused = True
                flag = False

                while True:
                    key = cv2.waitKey(1)

                    if key == 32:
                        paused = False
                        break
                    elif key == 27:
                        flag = True
                        break
                    elif updated:
                        break

                if flag:
                    break
            elif key == 13:
                current = "main"
                vc = vcs[current]
                vc.set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])
            elif 48 <= key <= 57:
                if metadata.get(chr(key)) is not None:
                    if metadata[chr(key)][index] != -1:
                        current = chr(key)
                        vc = vcs[current]

                        if index < len(metadata[chr(key)]):
                            if metadata[chr(key)][index] < 0:
                                current = "main"
                                vc = vcs[current]

                            vc.set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])
            elif key in [33, 64, 35, 36, 37, 94, 38, 42, 40, 41]:
                if metadata.get(letter2hotkey[key]) is not None:
                    if metadata[letter2hotkey[key]][index] != -1:
                        current = letter2hotkey[key]
                        vc = vcs[current]

                        if index < len(metadata[chr(key)]):
                            if metadata[chr(key)][index] < 0:
                                current = "main"
                                vc = vcs[current]

                            vc.set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])
            index += 1
        else:
            if current == "main":
                break
            else:
                current = "main"
                vc = vcs[current]
                vc.set(cv2.CAP_PROP_POS_FRAMES, metadata[current][index])

    if save:
        vw.release()

    for k, v in vcs.items():
        v.release()

    cv2.destroyAllWindows()
