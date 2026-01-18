import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import collections
import time

# ==========================================
# Global Configuration
# ==========================================
PREDICTION_AVG_WINDOW = 10
PREDICTION_THRESHOLD = 0.6  


MODEL_PATH = "models/emotion_model_deep_40.pth" 
MODEL_ARCHITECTURE = 'DeepEmotionCNN'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']
CAMERA_INDEX = 1 # for cv2


# ==========================================
# Model Definitions 
# ==========================================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepEmotionCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(DeepEmotionCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 -> 8
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ==========================================
# Inference Pipeline
# ==========================================

def load_model(path, architecture_name='EmotionCNN'):
    """
    Loads the model with weights from the specified path.
    """
    if architecture_name == 'EmotionCNN':
        model = EmotionCNN(num_classes=len(CLASSES))
    elif architecture_name == 'DeepEmotionCNN':
        model = DeepEmotionCNN(num_classes=len(CLASSES))
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Successfully loaded state_dict.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
        
    return model.to(DEVICE)

def preprocess_face(face_img):
    """
    Preprocesses the face image for the model.
    1. Grayscale
    2. Resize to 64x64
    3. ToTensor (scales to 0-1)
    """
    face_pil = Image.fromarray(face_img)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    face_tensor = transform(face_pil).unsqueeze(0)
    return face_tensor.to(DEVICE)

def main():
    model = load_model(MODEL_PATH, MODEL_ARCHITECTURE)
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {CAMERA_INDEX}. Trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return

    print("Starting webcam inference...")
    print("Press 'q' to quit.")

    prediction_history = collections.deque(maxlen=PREDICTION_AVG_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_roi = frame[y:y+h, x:x+w]
            
            input_tensor = preprocess_face(face_roi)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            prediction_history.append(probs)

            avg_probs = np.mean(prediction_history, axis=0)
            
            max_prob = np.max(avg_probs)
            predicted_class_idx = np.argmax(avg_probs)
            predicted_label = CLASSES[predicted_class_idx]

            if max_prob >= PREDICTION_THRESHOLD:
                text = f"{predicted_label}: {max_prob:.2f}"
                color = (0, 255, 0) # Green
            else:
                text = "Uncertain"
                color = (0, 0, 255) # Red

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
