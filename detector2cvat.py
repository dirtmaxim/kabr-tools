import os
import sys
import cv2
from lxml import etree
from tqdm import tqdm
from src.yolo import YOLOv8
from src.tracker import Tracker, Tracks
from src.object import Object
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

    yolo = YOLOv8(weights="yolov8x.pt", imgsz=3840, conf=0.5)

    for i, video in enumerate(videos):
        try:
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
            tracks = Tracks(max_disappeared=max_disappeared, interpolation=True)

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
            pbar = tqdm(total=size)

            while vc.isOpened():
                returned, frame = vc.read()

                if returned:
                    visualization = frame.copy()
                    predictions = yolo.forward(frame)
                    centroids = []
                    attributes = []

                    for prediction in predictions:
                        attribute = {}
                        centroids.append(YOLOv8.get_centroid(prediction[0]))
                        attribute["box"] = prediction[0]
                        attribute["confidence"] = prediction[1]
                        attribute["label"] = prediction[2]
                        attributes.append(attribute)

                    objects, colors = tracker.update(centroids)
                    objects = Object.object_factory(objects, centroids, colors, attributes=attributes)

                    for object in objects:
                        tracks.update(index, object)
                        Draw.track(visualization, tracks[object.object_id].centroids, object, 20)
                        Draw.bounding_box(visualization, object)
                        Draw.object_id(visualization, object)

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
            for track in tracks.values():
                xml_track = etree.Element("track", id=str(track.object_id), label=str(track.label), source="manual")

                for box_id, (box, frame_id, interpolated) in enumerate(
                        zip(track.boxes, track.indices, track.interpolated)):
                    # Mark the end of the track.
                    if box_id == len(track.boxes) - 1:
                        outside = "1"
                    else:
                        outside = "0"

                    # If box is not found, look for previous available box.
                    if box is None:
                        replacement = box_id

                        while box is None:
                            replacement -= 1
                            box = track.boxes[replacement]

                        interpolated = True

                    xml_box = etree.Element("box", frame=str(frame_id), outside=outside, occluded="0",
                                            keyframe=str(int(not interpolated)), xtl=f"{box[0]:.2f}",
                                            ytl=f"{box[1]:.2f}",
                                            xbr=f"{box[2]:.2f}", ybr=f"{box[3]:.2f}", z_order="0")

                    xml_track.append(xml_box)

                if len(track.boxes) > 0:
                    xml_page.append(xml_track)

            xml_document = etree.ElementTree(xml_page)
            xml_document.write(output_path, xml_declaration=True, pretty_print=True, encoding="utf-8")
        except:
            print("Something went wrong...")
