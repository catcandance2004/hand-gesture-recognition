# hand-gesture-recognition
Hand Gesture Recognition with Mediapipe and FNN

Dự án này thực hiện nhận diện cử chỉ tay bằng cách sử dụng Mediapipe để trích xuất đặc điểm từ hình ảnh và video, sau đó sử dụng một mạng nơ-ron tiến (FNN) để phân loại các cử chỉ. Dự án được thiết kế để hoạt động với bộ dữ liệu ASL Alphabet từ Kaggle, bao gồm các hình ảnh của bảng chữ cái ngôn ngữ ký hiệu Mỹ (A-Z).

Cấu trúc dự án

hand-gesture-recognition/
│
├── data/
│   ├── asl_dataset/                # Thư mục chứa dữ liệu thô từ Kaggle
│   ├── processed_data/             # Thư mục chứa dữ liệu đã qua xử lý
│   │   ├── landmarks_train.csv     # Landmarks đã trích xuất từ tập huấn luyện
│   │   ├── landmarks_test.csv      # Landmarks đã trích xuất từ tập kiểm tra
│   │   ├── train_labels.npy        # Nhãn của tập huấn luyện
│   │   └── test_labels.npy         # Nhãn của tập kiểm tra
│   └── models/                     # Thư mục lưu mô hình đã huấn luyện
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py         # Script xử lý dữ liệu
│   ├── model.py                    # Script định nghĩa và huấn luyện mô hình
│   ├── utils.py                    # Các hàm tiện ích
│   └── real_time_recognition.py    # Ứng dụng nhận dạng thời gian thực
│
├── notebooks/
│   ├── data_exploration.ipynb      # Khám phá dữ liệu
│   ├── model_training.ipynb        # Huấn luyện mô hình
│   └── evaluation.ipynb            # Đánh giá mô hình
│
└── README.md                       # Tài liệu hướng dẫn

Những gì đã làm được

Dưới đây là phân tích chi tiết về những gì đã thực hiện trong từng thành phần của dự án, bao gồm cách sử dụng và chức năng của chúng.

1. Thư mục data/
asl_dataset/: bộ dữ liệu ASL Alphabet từ Kaggle, chứa các hình ảnh thô của bảng chữ cái ASL (A-Z), mỗi lớp được tổ chức trong thư mục con riêng biệt (ví dụ: 'A', 'B', ...).
Chức năng: Cung cấp dữ liệu đầu vào ban đầu để xử lý và huấn luyện mô hình.
Sử dụng: Được truy cập bởi script data_preparation.py để trích xuất đặc điểm.

processed_data/
Tạo các file landmarks_train.csv và landmarks_test.csv chứa tọa độ landmarks (điểm mốc) được trích xuất từ hình ảnh bằng Mediapipe.
Tạo các file train_labels.npy và test_labels.npy lưu nhãn tương ứng dưới dạng mảng NumPy.
Chức năng: Lưu trữ dữ liệu đã qua xử lý, sẵn sàng cho việc huấn luyện và đánh giá mô hình.
Sử dụng: Được tải bởi model.py để huấn luyện và evaluation.ipynb để đánh giá.

models/
Đã làm được: Lưu trữ mô hình FNN đã huấn luyện (ví dụ: fnn_model.h5) và mô hình TFLite tối ưu cho nhận dạng thời gian thực (ví dụ: keypoint_classifier.tflite).
Chức năng: Cung cấp các mô hình đã huấn luyện để tái sử dụng trong nhận dạng hoặc đánh giá.
Sử dụng: Được tải bởi real_time_recognition.py và evaluation.ipynb.

2. Thư mục src/

data_preparation.py
Tải hình ảnh và nhãn từ asl_dataset/.
Sử dụng Mediapipe để trích xuất landmarks từ mỗi hình ảnh.
Chuẩn hóa và làm phẳng dữ liệu landmarks, sau đó lưu vào processed_data/.

Chức năng: Chuẩn bị dữ liệu thô thành định dạng phù hợp cho mô hình FNN.

Sử dụng: Chạy độc lập để tạo dữ liệu xử lý:

python src/data_preparation.py --data_path data/asl_dataset --save_path data/processed_data

model.py
Định nghĩa kiến trúc FNN với các lớp Dense, Dropout, và BatchNormalization để chống overfitting.
Huấn luyện mô hình trên dữ liệu từ processed_data/ với các callback như EarlyStopping và ReduceLROnPlateau.
Lưu mô hình đã huấn luyện vào models/.

Chức năng: Xây dựng và huấn luyện mô hình phân loại cử chỉ tay.

Sử dụng: Chạy để huấn luyện mô hình:
python src/model.py --train_data data/processed_data/landmarks_train.csv --train_labels data/processed_data/train_labels.npy --val_data data/processed_data/landmarks_val.csv --val_labels data/processed_data/val_labels.npy --save_path models/fnn_model.h5

utils.py
Đã làm được:
Triển khai các hàm tiện ích như load_processed_data() để tải dữ liệu, encode_labels() để mã hóa nhãn, và plot_confusion_matrix() để trực quan hóa kết quả.
Chức năng: Hỗ trợ các tác vụ lặp lại trong dự án, tăng tính tái sử dụng của mã nguồn.
Sử dụng: Được import và gọi bởi các script/notebook khác.

real_time_recognition.py
Đã làm được:
Sử dụng camera để chụp khung hình liên tục.
Trích xuất landmarks bằng Mediapipe, dự đoán cử chỉ bằng mô hình TFLite, và hiển thị kết quả trên màn hình.

Chức năng: Triển khai ứng dụng nhận dạng cử chỉ tay thời gian thực.

Sử dụng: Chạy để demo nhận dạng:

python src/real_time_recognition.py

3. Thư mục notebooks/ (chưa chạy được, help)

data_exploration.ipynb
Đã làm được:
Hiển thị mẫu hình ảnh từ các lớp khác nhau.
Kiểm tra phân phối lớp và trực quan hóa landmarks trên hình ảnh mẫu.

Chức năng: Khám phá dữ liệu để hiểu đặc điểm và chất lượng dataset.

Sử dụng: Mở và chạy từng cell trong Jupyter Notebook.


model_training.ipynb
Đã làm được:
Tải dữ liệu đã xử lý, thử nghiệm huấn luyện FNN, và vẽ biểu đồ lịch sử huấn luyện.

Chức năng: Phát triển và tinh chỉnh mô hình một cách tương tác.

Sử dụng: Chạy trong Jupyter Notebook để thử nghiệm.


evaluation.ipynb
Đã làm được:
Tải mô hình và dữ liệu kiểm tra, tính toán các chỉ số (độ chính xác, recall, f1-score), và vẽ ma trận nhầm lẫn.
Chức năng: Đánh giá hiệu suất mô hình trên tập kiểm tra.

Sử dụng: Chạy trong Jupyter Notebook để phân tích kết quả.

Những gì cần làm tiếp theo
1. Hoàn thiện mã nguồn (src/)

Tích hợp các script trong src/ thành một pipeline hoàn chỉnh: từ xử lý dữ liệu (data_preparation.py), huấn luyện mô hình (model.py), đến nhận dạng thời gian thực (real_time_recognition.py).
Thêm tính năng lưu trữ video hoặc hình ảnh khi nhận dạng thành công trong real_time_recognition.py.
Tối ưu hóa hiệu suất của real_time_recognition.py bằng cách giảm độ trễ xử lý khung hình.

2. Chuẩn bị slide
Tạo slide trình bày gồm các phần:

Giới thiệu: Mục tiêu và ý nghĩa của dự án.
Phương pháp: Quy trình từ xử lý dữ liệu, huấn luyện mô hình, đến nhận dạng thời gian thực.
Kết quả: Độ chính xác, ma trận nhầm lẫn, và demo video.
Kết luận: Những gì đã đạt được và hướng phát triển.

Bao gồm hình ảnh minh họa (landmarks, giao diện nhận dạng) và biểu đồ từ notebooks/.

3. Viết báo cáo
Soạn thảo báo cáo chi tiết với các mục:
Tổng quan: Giới thiệu vấn đề và cách tiếp cận.
Phương pháp: Mô tả Mediapipe, FNN, và quy trình thực hiện.
Thí nghiệm: Kết quả định lượng (độ chính xác, f1-score) và định tính (demo).
Thách thức: Các vấn đề gặp phải (ví dụ: chất lượng dữ liệu, tốc độ xử lý).
Kết luận và hướng mở rộng: Tóm tắt và đề xuất cải tiến.
Đính kèm biểu đồ, bảng số liệu, và hình ảnh từ dự án.

4. Tối ưu hóa và mở rộng
Thử nghiệm các kiến trúc mô hình khác (CNN, LSTM) để so sánh hiệu suất với FNN.

Mở rộng dataset hoặc thử nghiệm với dataset khác để kiểm tra tính tổng quát.

Tối ưu hóa mô hình TFLite để triển khai trên thiết bị di động.

5. Hướng dẫn cài đặt và chạy dự án

Thư viện: numpy, pandas, opencv-python, mediapipe, tensorflow, scikit-learn, matplotlib, seaborn

Cài đặt
Clone repository:

git clone https://github.com/your-repo/hand-gesture-recognition.git
cd hand-gesture-recognition



Cài đặt thư viện:
pip install -r requirements.txt

Chạy dự án

Chuẩn bị dữ liệu:
python src/data_preparation.py --data_path data/asl_dataset --save_path data/processed_data

Huấn luyện mô hình:

python src/model.py --train_data data/processed_data/landmarks_train.csv --train_labels data/processed_data/train_labels.npy --val_data data/processed_data/landmarks_val.csv --val_labels data/processed_data/val_labels.npy --save_path models/fnn_model.h5