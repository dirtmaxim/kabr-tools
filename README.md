# imageomics-tools

This repository contains tool for working with data for annotating animal behavior.

![](https://user-images.githubusercontent.com/11778655/220999935-dd190898-7cbc-4e83-acfd-ce9d2b7d24e5.png)

detector2cvat.py:\
Detect objects with Ultralytics YOLO detections, apply SORT tracking and convert tracks to CVAT format.

```
python detector2cvat.py path_to_videos path_to_save
```

cvat2ultralytics.py:\
Convert CVAT annotations to Ultralytics YOLO dataset.

```
python cvat2ultralytics.py path_to_videos path_to_annotations dataset_name [skip_frames]
```

tracks_extractor.py:\
Extract mini-scenes from CVAT tracks.

```
python tracks_extractor.py path_to_videos path_to_annotations [tracking]
```

player.py:\
Player for track and behavior observation.

```
python player.py path_to_folder [save]
```


cvat2slowfast.py:\
Convert CVAT annotations to SlowFast Charades dataset.

```
python cvat2slowfast.py path_to_mini_scenes dataset_name [zebra, giraffe]
```