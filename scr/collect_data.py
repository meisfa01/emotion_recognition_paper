import cv2
import os
import time
import uuid
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']
CAMERA_INDEX = 1 
OUTPUT_DIR = "test_data"

def create_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for cls in CLASSES:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

def record_frames(emotion, cap):
    frames = []
    print(f"\nReady to record for emotion: {emotion}")
    print("Press 'r' to START recording.")
    print("Press 'q' to STOP recording.")
    
    recording = False
    print("Waiting for start...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        display_frame = frame.copy()
        
        if recording:
            frames.append(frame.copy())
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1) # Red dot for recording
            cv2.putText(display_frame, f"Recording {emotion}: {len(frames)} frames", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press 'r' to start", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Recording', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                print("Recording started...", flush=True)
                recording = True
                frames = [] # Reset frames on start
        elif key == ord('q'):
            if recording:
                print("Recording stopped.", flush=True)
                break
            else:
                # If not recording, q quits the whole process or just this emotion?
                # Let's say it finishes this session.
                break
    
    return frames

def process_frames(frames, emotion):
    if not frames:
        print("No frames recorded.")
        return

    print(f"Processing {len(frames)} frames for emotion '{emotion}'...")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    save_count = 0
    
    
    for i, frame in enumerate(tqdm(frames)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Find largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            face_roi = gray[y:y+h, x:x+w]
            
            filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(OUTPUT_DIR, emotion, filename)
            cv2.imwrite(filepath, face_roi)
            save_count += 1
            
    print(f"Saved {save_count} images to {os.path.join(OUTPUT_DIR, emotion)}")

def main():
    create_dirs()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {CAMERA_INDEX}. Trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return

    print("========================================")
    print("Offline Data Collection Script")
    print("========================================")
    
    while True:
        print("\nSelect Emotion to Record:")
        for i, cls in enumerate(CLASSES):
            print(f"{i + 1}. {cls}")
        print("q. Quit")
        
        choice = input("Enter choice (1-5 or q): ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(CLASSES):
                emotion = CLASSES[idx]
                
                # Record Phase
                frames = record_frames(emotion, cap)
                
                # Process Phase
                if frames:
                    process_frames(frames, emotion)
                
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
