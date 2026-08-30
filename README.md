# Face Emotion Recognition System (FER)

A deep learning-based "Face Emotion Recognition System" developed as a Computer Engineering graduation project.

The system analyzes facial expressions from video data and classifies them into eight emotion categories using a CNN + LSTM architecture. The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

## Overview

Facial expressions contain important information about a person's emotional state. This project aims to automatically detect and classify human emotions from video data using computer vision and deep learning techniques.

The implemented system processes video frames, extracts visual features using **ResNet-18**, models temporal information using an **LSTM**, and predicts one of eight emotion classes.

The project was developed using Python and  PyTorch

## Emotions

The system recognizes the following eight emotion categories:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

## Model Architecture

The project uses a CNN-LSTM architecture:

```text
Input Video
     │
     ▼
Frame Sampling
     │
     ▼
Image Preprocessing
     │
     ▼
Face Detection
(Haar Cascade - optional)
     │
     ▼
ResNet-18
Visual Feature Extraction
     │
     ▼
LSTM
Temporal Feature Learning
     │
     ▼
Fully Connected Layer
     │
     ▼
8 Emotion Classes
```

### CNN

A **ResNet-18** network is used as the CNN backbone.

The final classification layer of ResNet-18 is replaced with an identity layer so that the network produces a **512-dimensional feature vector** for each frame.

### LSTM

The extracted frame features are passed to an LSTM to learn temporal relationships between video frames.

Main parameters:

* Input size: 512
* Hidden size: 256
* LSTM layers: 1
* Dropout: 0.3
* Bidirectional: No

### Classification

The final LSTM representation is passed through a fully connected layer to classify the input into eight emotion categories.

## Dataset

The project uses the RAVDESS dataset.

RAVDESS contains audio-visual emotional recordings performed by professional actors. The project uses its video data for facial emotion recognition.

According to the project report, the dataset contains:

* 24 actors
* 12 male actors
* 12 female actors
* 8 emotion categories
* 1,440 video clips

The RAVDESS filename format is used to automatically determine the emotion label.

The third field of the filename represents the emotion code.

## Data Processing

For each video:

1. Video frames are loaded using OpenCV.
2. A fixed number of frames is sampled.
3. Frames are converted from BGR to RGB.
4. Frames are resized to `224 × 224`.
5. Image normalization is applied.
6. Haar Cascade can optionally be used for face detection and cropping.
7. The processed frames are passed to the CNN-LSTM model.

The default number of frames used by the training system is:

```text
32 frames per video
```

## Training

The model is trained using:

* Framework: PyTorch
* CNN: ResNet-18
* Temporal model: LSTM
* Loss function: Cross Entropy Loss
* Optimizer: Adam
* Learning rate: `1e-4`
* Batch size: `4`
* Number of epochs: `10`
* Validation split: `20%`
* Image size: `224 × 224`

The best-performing model is saved as:

```text
checkpoints/best_model.pth
```

## Evaluation

The project report evaluates the model using a validation set.

The reported results are:

```text
Validation samples: 284
Correct predictions: 204
Accuracy: 71.83%
```

> Note: The project uses the validation set for evaluation and does not use a separate independent test dataset.

## Project Structure

```text
face-emotion-recognition/
│
├── src/
│   ├── inference.py
│   ├── data.py
│   ├── model.py
│   └── train.py
│
├── checkpoints/
│   └── best_model.pth
│
├── .gitignore
├── README.md
├── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
pip install -r requirements.txt
```

## Requirements

The project uses the following main Python libraries:

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* scikit-learn
* tqdm

See `requirements.txt` for the required packages.

## Training the Model

Place the RAVDESS video dataset on your computer and provide its directory to the training script.

Example:

```bash
python src/train.py --data-dir "path/to/RAVDESS"
```

Optional face detection:

```bash
python src/train.py --data-dir "path/to/RAVDESS" --face-crop
```

The trained model will be saved in:

```text
checkpoints/best_model.pth
```

## Running Emotion Recognition

The inference program can analyze a video using the trained checkpoint.

Controls

During video analysis:

Q / ESC     Quit
P           Pause / Resume
Space       Step through frames when paused
R           Reset analysis

