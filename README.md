# Real-Time Human Action Recognition Using Deep Learning

A real-time Human Action Recognition (HAR) system using a hybrid
MobileNetV2 and Bidirectional LSTM architecture to recognize human
activities from video sequences.

## Overview

The system combines MobileNetV2 for spatial feature extraction and
Bidirectional LSTM for learning temporal patterns from sequences of
video frames.

The system is designed for real-time action recognition using webcam
input and OpenCV.

## Actions Recognized

The model recognizes six human activities:

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

## Methodology

1. Dataset collection
2. Frame extraction
3. Image resizing and normalization
4. Data augmentation
5. Spatial feature extraction using MobileNetV2
6. Temporal sequence learning using BiLSTM
7. Softmax-based action classification
8. Real-time prediction using OpenCV and webcam input

## Model Architecture

MobileNetV2 → BiLSTM → Dense → Softmax

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- MobileNetV2
- BiLSTM
- NumPy
- Scikit-learn

## Dataset

The model was trained using the KTH Human Action Dataset.

The dataset contains six action classes:

Boxing, Handclapping, Handwaving, Jogging, Running, and Walking.

The original dataset is not included in this repository.

## Results

| Model                            | Accuracy |
| MobileNetV2 (CNN)                | 84.26%   |
| MobileNetV2 + LSTM               | 87.04%   |
| MobileNetV2 + BiLSTM + Attention | 93.06%   |
| MobileNetV2 + BiLSTM             | 95.37%   |

## Real-Time Implementation

The trained model can be used with webcam input to recognize human
actions in real time using OpenCV.

## Project Structure

preprocessing/
training/
models/
Results/
real_time.py
realtim.py
video_test_input.py
requirements.txt
