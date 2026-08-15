# Real-Time Human Action Recognition Using Deep Learning

A real-time Human Action Recognition (HAR) system using a hybrid
MobileNetV2 and Bidirectional LSTM (BiLSTM) architecture to recognize
human activities from video sequences.

## Overview

The system combines MobileNetV2 for spatial feature extraction and
Bidirectional LSTM for learning temporal patterns from sequences of
video frames.

The trained model can be used for real-time human action recognition
using webcam input and OpenCV.

## Actions Recognized

The system recognizes six human actions:

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
4. Data preprocessing and augmentation
5. Spatial feature extraction using MobileNetV2
6. Temporal sequence learning using BiLSTM
7. Softmax-based action classification
8. Model evaluation
9. Real-time prediction using OpenCV

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

This project uses the **KTH Human Action Dataset**.

The dataset contains six human action classes performed by multiple
subjects in different scenarios.

The original dataset is **not included in this repository because of
its large size**.

### Download Dataset

Download the KTH Action Dataset from the official KTH website:

https://www.csc.kth.se/cvap/actions/

After downloading the dataset, organize the videos according to the
action classes used by the preprocessing scripts.

The dataset should be placed locally in:

```text
data/
├── boxing/
├── handclapping/
├── handwaving/
├── jogging/
├── running/
└── walking/

The extracted video frames can then be generated using the preprocessing
scripts and stored in:

data_frames/
├── boxing/
├── handclapping/
├── handwaving/
├── jogging/
├── running/
└── walking/

### Project Structure
real-time-human-action-recognition/
│
├── preprocessing/
│   ├── dataset_builder.py
│   ├── dataset_builder_subject.py
│   └── frame_extractor.py
│
├── training/
│   ├── train_model.py
│   ├── Train_cnn_model.py
│   ├── train_cnn_lstm_model.py
│   ├── train_attention_model.py
│   ├── Train_subject_model.py
│   └── evaluate_models.py
│
├── models/
│   └── best_models.keras
│
├── Results/
│
├── real_time.py
├── realtim.py
├── video_test_input.py
├── requirements.txt
└── README.md

## Installation

Clone the repository:

git clone https://github.com/Pagilla-Joshna/real-time-har.git

Navigate to the project directory:

cd real-time-human-action-recognition

Install the required packages:

pip install -r requirements.txt

### Preprocessing

After downloading and placing the KTH dataset in the data/ directory,
run the frame extraction script:

python preprocessing/frame_extractor.py

The extracted frames can then be used to build the training dataset:

python preprocessing/dataset_builder.py

###Training

To train the proposed MobileNetV2 + BiLSTM model:

python training/train_model.py

The training process generates the trained model and saves the required
model files for evaluation and inference.

### Model Evaluation

To evaluate the trained model:

python training/evaluate_models.py

Evaluation can include metrics such as accuracy, precision, recall,
F1-score and confusion matrix.

### Real-Time Recognition

After training the model, run the real-time recognition program:

python real_time.py

The system uses webcam input and OpenCV to recognize human actions in
real time.

###Video Testing

For testing the model using a video file:

python video_test_input.py
Results

The proposed MobileNetV2 + BiLSTM model achieved a reported accuracy of
95.37% in the project evaluation.

Model	                            Accuracy
MobileNetV2 (CNN)	                 84.26%
MobileNetV2 + LSTM	               87.04%
MobileNetV2 + BiLSTM + Attention	 93.06%
MobileNetV2 + BiLSTM (Proposed)	   95.37%

### Future Improvements
Support for additional human action classes
Improved real-time inference performance
Larger and more diverse datasets
Edge-device deployment
Web-based application deployment
