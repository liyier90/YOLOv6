# PeekingDuck Familiarization Task Instructions

## Preparing the repo
1. Fork the repository, clone it and navigate to its directory
   ```
   $ git clone https://github.com/<YOUR_NAME>/YOLOv6.git && cd YOLOv6
   ```
2. Download the COCO dataset
   ```
   $ bash data/get_coco_dataset.sh
   ```
   *Note: The COCO dataset downloaded by the script deviates from the official
   COCO dataset as it contains an extra "labels" folder (obtained from the
   YOLOv5 repo).*
3. Create a `weights` directory and download the
   [YOLOv6-n weights](https://storage.googleapis.com/peekingduck/familiarization/yolov6n.pt)
   into the directory.

You should have the following directory structure at this point:
```
.
+---coco/
|   +---annotations/
|   +---images/
|   \---labels/
\---YOLOv6/
    +---...
    \---weights/
        \---yolov6n.pt
```
*Note: The "coco/images/" directory contains only "val2017" while the
"coco/labels/" contains both "train2017" and "val2017". This is fine as we are
only concerned with the "val2017" subset for the purpose of this exercise.*

## Evaluation
1. Install the necessary requirements.
2. Evaluate mAP on the COCO val2017 dataset
   ```
   python tools/eval.py --data data/coco.yaml --batch 32 --weights weights/yolov6n.pt 
   ```