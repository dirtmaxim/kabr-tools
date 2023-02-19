# imageomics-tools

This repository contains tool for working with data for annotating animal behavior.

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
python player.py path_to_video [save]
```

cvat2ultralytics.py:\
Convert CVAT annotations to Ultralytics YOLO dataset.

```
python cvat2ultralytics.py path_to_videos path_to_annotations dataset_name [skip_frames]
```