import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Path to TFLite model
MODEL_PATH = "model/keypoint_classifier/keypoint_classifier.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of classes (e.g., A-Z)
CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]

def pre_process_landmark(landmark_list):
    """Normalizes landmark coordinates relative to wrist."""
    if not landmark_list:
        return None
    wrist = np.array(landmark_list[0])
    landmarks = np.array(landmark_list)
    normalized = landmarks - wrist
    flattened = normalized.flatten()
    return flattened / np.linalg.norm(flattened)

def predict_gesture(landmark_array):
    """Predicts gesture using the TFLite model."""
    interpreter.set_tensor(input_details[0]['index'], np.array([landmark_array], dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Queue for smoothing predictions
    prediction_queue = deque(maxlen=5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract and preprocess landmarks
                landmark_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                processed_landmark = pre_process_landmark(landmark_list)
                
                if processed_landmark is not None:
                    # Predict gesture
                    gesture_id = predict_gesture(processed_landmark)
                    prediction_queue.append(gesture_id)
                    
                    # Get most common gesture in queue
                    most_common_gesture = max(set(prediction_queue), key=prediction_queue.count)
                    gesture_label = CLASSES[most_common_gesture]
                    
                    # Display result
                    cv2.putText(frame, f"Gesture: {gesture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Real-time Hand Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()