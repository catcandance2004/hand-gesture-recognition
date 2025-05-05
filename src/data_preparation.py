import os
import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool
from tqdm import tqdm

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def load_images_and_labels(data_path):
    """Load images and labels from dataset directory."""
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

def extract_landmarks(image):
    """Extract hand landmarks from an image using Mediapipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return None

def preprocess_landmarks(landmark_list):
    """Normalize and flatten landmarks relative to wrist."""
    if not landmark_list:
        return None
    wrist = np.array(landmark_list[0])  # Wrist is landmark 0
    landmarks = np.array(landmark_list)
    normalized = landmarks - wrist  # Relativize to wrist
    flattened = normalized.flatten()
    return flattened / np.linalg.norm(flattened)  # Normalize magnitude

def process_image(args):
    """Helper function for multiprocessing."""
    img, label = args
    landmarks = extract_landmarks(img)
    if landmarks:
        processed = preprocess_landmarks(landmarks)
        if processed is not None:
            return processed, label
    return None, None

def save_processed_data(landmarks, labels, save_path):
    """Save processed landmarks and labels to files."""
    np.save(os.path.join(save_path, 'landmarks.npy'), landmarks)
    np.save(os.path.join(save_path, 'labels.npy'), labels)

def prepare_data(data_path, save_path):
    """Main function to prepare data."""
    images, labels = load_images_and_labels(data_path)
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_image, zip(images, labels)), total=len(images)))
    
    valid_landmarks = []
    valid_labels = []
    for landmarks, label in results:
        if landmarks is not None:
            valid_landmarks.append(landmarks)
            valid_labels.append(label)
    
    valid_landmarks = np.array(valid_landmarks)
    valid_labels = np.array(valid_labels)
    save_processed_data(valid_landmarks, valid_labels, save_path)
    print(f"Processed data saved to {save_path}")

if __name__ == "__main__":
    data_path = "path/to/asl_alphabet_dataset"
    save_path = "path/to/save_processed_data"
    os.makedirs(save_path, exist_ok=True)
    prepare_data(data_path, save_path)