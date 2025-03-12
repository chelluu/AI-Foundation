import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time

# Load trained CNN model
model_path = r"C:\Users\User\Desktop\Python\HandMotionDetectionAI\best_model.h5"
model = tf.keras.models.load_model(model_path)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# Define class labels (update this according to your dataset)
class_labels = ["Pause", "Play", "Rewind", "Skip"]

def preprocess_hand_landmarks(hand_landmarks, img_shape):
    """Extracts hand landmark coordinates and normalizes them for CNN input."""
    h, w, _ = img_shape
    landmark_points = []
    
    for lm in hand_landmarks.landmark:
        landmark_points.append(lm.x)
        landmark_points.append(lm.y)

    # Convert to NumPy array and reshape to match CNN input
    landmark_array = np.array(landmark_points).reshape(1, -1)
    return landmark_array

def classify_gesture(frame):
    """Preprocess the captured frame and predict gesture using CNN."""
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

def preprocess_frame(frame):
    """Resize frame while maintaining aspect ratio and adding padding."""
    target_size = (64, 64)

    # Get original dimensions
    h, w, _ = frame.shape
    aspect_ratio = w / h

    # Compute new width & height while maintaining aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_w = 64
        new_h = int(64 / aspect_ratio)
    else:  # Taller than wide
        new_h = 64
        new_w = int(64 * aspect_ratio)

    # Resize while keeping aspect ratio
    resized = cv2.resize(frame, (new_w, new_h))

    # Create a black canvas and center the resized image
    padded = np.zeros((64, 64, 3), dtype=np.uint8)
    pad_x = (64 - new_w) // 2
    pad_y = (64 - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Normalize and reshape for CNN
    padded = padded.astype("float32") / 255.0
    return np.expand_dims(padded, axis=0)  # Add batch dimension

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for better mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Predict gesture
                gesture = classify_gesture(frame)  # Use the whole frame, not landmarks

                # Display the detected gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Perform actions based on the detected gesture
                if gesture == "Play":
                    pyautogui.press("space")  # Play/Pause
                    time.sleep(1)
                elif gesture == "Pause":
                    pyautogui.press("space")  # Play/Pause
                    time.sleep(1)
                elif gesture == "Skip":
                    pyautogui.press("right")  # Skip forward
                    time.sleep(1)
                elif gesture == "Rewind":
                    pyautogui.press("left")  # Rewind
                    time.sleep(1)

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()