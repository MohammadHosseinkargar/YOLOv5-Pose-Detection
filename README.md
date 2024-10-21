# YOLOv5 Pose Detection

This project implements a real-time human pose detection system using the YOLOv5 model for object detection and MediaPipe for 3D pose estimation. Additionally, it leverages OpenAI's GPT-3 to generate textual descriptions based on recognized actions and movements.

## Features

- **Object Detection**: Utilizes the YOLOv5 model to detect humans in real-time.
- **3D Pose Estimation**: Employs MediaPipe to estimate body landmarks and joint angles in a 3D space.
- **Action Recognition**: Implements a basic rule-based system to recognize actions based on body angles.
- **Text Generation**: Generates narrative descriptions of detected actions using the OpenAI GPT-3 API.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- MediaPipe
- OpenAI API
- Pandas
- NumPy

## Installation

To install the necessary dependencies, you can use pip:

```bash
pip install opencv-python torch mediapipe openai pandas numpy
