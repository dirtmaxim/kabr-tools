# imageomics-tools

This repository contains tool for working with data for annotating animal behavior.


![](https://user-images.githubusercontent.com/11778655/220793356-60ad0253-865c-4de0-9ee7-9c08a3fd3927.png)

tracks_extractor.py:\
Extract mini-scenes from CVAT tracks.

```
python tracks_extractor.py path_to_videos path_to_annotations [tracking]
```

actions_extractor.py:\
Extract actions from CVAT annotations.

```
python actions_extractor.py path_to_annotations
```

player.py:\
Player for track and behavior observation.

```
python player.py path_to_folder [save]
```

cvat2ultralytics.py:\
Convert CVAT annotations to Ultralytics YOLO dataset.

```
python cvat2ultralytics.py path_to_videos path_to_annotations dataset_name [skip_frames]
```

detector2cvat.py:\
Detect objects with Ultralytics YOLO detections, apply SORT tracking and convert tracks to CVAT format.

```
python detector2cvat.py path_to_videos path_to_save
```
