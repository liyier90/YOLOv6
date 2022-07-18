#!/usr/bin/env bash

# Download dataset
wget https://storage.googleapis.com/peekingduck/familiarization/coco-yolov6.zip

# Extract and clean up
unzip coco-yolov6.zip -d .. && rm coco-yolov6.zip