import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from typing import List, Tuple
import csv
import os

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Đường dẫn mô hình và file dữ liệu
MODEL_PATH = "model/keypoint_classifier/keypoint_classifier.tflite"
CSV_PATH = "model/keypoint_classifier/keypoint.csv"

# Tải mô hình TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def pre_process_landmark(landmark_list: List[float]) -> np.ndarray:
    """Chuẩn hóa tọa độ landmark: relativize và normalize."""
    temp_landmark = np.array(landmark_list, dtype=np.float32)
    base_x, base_y = temp_landmark[0], temp_landmark[1]  # Gốc tại cổ tay (landmark 0)
    # Relativize: trừ đi gốc
    temp_landmark[0::2] -= base_x  # x coords
    temp_landmark[1::2] -= base_y  # y coords
    # Normalize: chia cho giá trị tuyệt đối lớn nhất
    max_val = np.max(np.abs(temp_landmark))
    if max_val != 0:
        temp_landmark /= max_val
    return temp_landmark

def calc_landmark_list(image: np.ndarray, landmarks) -> List[float]:
    """Trích xuất danh sách tọa độ landmark từ kết quả Mediapipe."""
    image_height, image_width = image.shape[:2]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.extend([landmark_x, landmark_y])
    return landmark_point

def log_keypoint(label: int, landmark_list: List[float]) -> None:
    """Ghi dữ liệu landmark vào CSV."""
    with open(CSV_PATH, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

def predict_gesture(landmark_array: np.ndarray) -> int:
    """Dự đoán cử chỉ từ mô hình TFLite."""
    interpreter.set_tensor(input_details[0]['index'], np.array([landmark_array], dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    
    mode = 0  # 0: Dự đoán, 1: Ghi dữ liệu
    label = 0  # Nhãn mặc định
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Xử lý ảnh với Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmark
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Trích xuất và chuẩn hóa landmark
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed_landmark = pre_process_landmark(landmark_list)
                
                if mode == 1:  # Ghi dữ liệu
                    log_keypoint(label, processed_landmark)
                    cv2.putText(frame, f"Logging: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:  # Dự đoán
                    gesture_id = predict_gesture(processed_landmark)
                    cv2.putText(frame, f"Gesture: {gesture_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị frame
        cv2.imshow("Hand Gesture Recognition", frame)
        
        # Điều khiển chế độ
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Thoát
            break
        elif key == ord('k'):  # Bật/tắt chế độ ghi dữ liệu
            mode = 1 - mode
        elif mode == 1 and ord('0') <= key <= ord('9'):  # Chọn nhãn 0-9
            label = key - ord('0')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()