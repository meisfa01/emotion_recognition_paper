# Human Face Emotion Detection

This project implements a Real-time Human Face Emotion Detection system using Deep Learning (CNN). It captures video from a webcam, detects faces, and classifies the emotion into one of 5 categories: Angry, Fear, Happy, Sad, Surprise.

The final paper describing the project details is included in this repository.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requrements.txt
    ```

## Usage

### 1. Real-time Inference

To run the emotion detection using the pre-trained model:

```bash
python detect_face_emotion/inference.py
```

This will open a window showing the webcam feed with the predicted emotion and confidence score. Press 'q' to quit.

*Note: The script attempts to use camera index 1 by default. If it fails, it will try index 0.*

### 2. Data Collection

To collect your own data for training or testing:

```bash
python detect_face_emotion/collect_data.py
```

Follow the interactions in the terminal to select an emotion label and record frames. The script extracts faces and saves them to `detect_face_emotion/test_data`.

### 3. Training

The training pipeline and hyperparameter optimization are implemented in Jupyter Notebooks:

-   `detect_face_emotion/prepare_data_and_train.ipynb`: Main notebook for data preparation and model training.
-   `detect_face_emotion/hyperparameter_optimization.ipynb`: Notebook for optimizing model parameters.
