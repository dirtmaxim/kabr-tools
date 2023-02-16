# imageomics-tools
This repository contains tool for working with data for annotating animal behavior.

extractor.py:\
Extract mini-scenes from CVAT tracks.
```
python extractor.py path_to_video path_to_annotation [tracking]
```

cvat2ultralytics.py:\
Convert CVAT annotations to Ultralytics COCO dataset.
```
python cvat2ultralytics.py path_to_videos path_to_annotations dataset_name [skip_frames]
```

counter.py:\
List all video files in a folder with corresponding length.
```
python counter.py path_to_folder
```