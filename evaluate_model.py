import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
import argparse

from inference import load_model, preprocess_face, CLASSES, DEVICE, MODEL_PATH, MODEL_ARCHITECTURE

def evaluate(data_dir, model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, MODEL_ARCHITECTURE)
    if model is None:
        return

    model.eval()

    y_true = []
    y_pred = []

    print(f"Evaluating on data in {data_dir}...")

    # Iterate through each class folder
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(files)} images for class '{class_name}'")

        for f in files:
            img_path = os.path.join(class_dir, f)
            try:
                image = Image.open(img_path).convert('L') 
                image_np = np.array(image)
                
                input_tensor = preprocess_face(image_np)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_class_idx = np.argmax(probs)
                    predicted_label = CLASSES[predicted_class_idx]

                y_true.append(class_name)
                y_pred.append(predicted_label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not y_true:
        print("No images found for evaluation.")
        return

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "="*30)
    print(f"Accuracy: {accuracy:.4f}")
    print("="*30)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, labels=CLASSES))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save plot
    output_plot_path = os.path.join("..", "report", "figures", "confusion_matrix.png")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    plt.savefig(output_plot_path)
    print(f"\nConfusion matrix saved to {output_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Model")
    parser.add_argument("--data_dir", type=str, default="test_data", help="Path to the test data directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model file")
    
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.model_path)
