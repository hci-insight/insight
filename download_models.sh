#!/usr/bin/env bash
# Download the face-detection models used by shot_capture_experiment.py.
# Run once after cloning:  bash download_models.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$DIR"

echo "Downloading models into: $DIR"

# MediaPipe Face Detection (Tasks API, BlazeFace short-range) -- ~230 KB
curl -fsSL -o "$DIR/blaze_face_short_range.tflite" \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

# OpenCV YuNet face detector (ONNX) -- ~230 KB
curl -fsSL -o "$DIR/face_detection_yunet_2023mar.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "Done:"
ls -la "$DIR"
