# 👕 Phân Loại Quần Áo với Deep Learning và MLflow
Dự án này sử dụng Deep Learning để phân loại các hình ảnh quần áo vào các danh mục khác nhau như áo thun, quần, váy, v.v., đồng thời tích hợp MLflow để theo dõi và quản lý quá trình huấn luyện mô hình, giúp dễ dàng so sánh các mô hình và tham số huấn luyện.

🎯 Mục Tiêu Dự Án
Phân loại hình ảnh quần áo: Dự án này nhằm mục đích xây dựng mô hình học sâu (Deep Learning) để phân loại hình ảnh quần áo vào các loại như áo thun, quần, váy, giày, v.v.

Sử dụng CNN: Mạng nơ-ron tích chập (Convolutional Neural Network - CNN) được sử dụng để học và phân loại các đặc trưng trong hình ảnh.

Theo dõi với MLflow: MLflow được tích hợp để theo dõi quá trình huấn luyện mô hình, ghi lại các tham số, độ chính xác, và các chỉ số hiệu suất khác, giúp dễ dàng quản lý và so sánh các thí nghiệm.

📁 Cấu trúc Thư Mục
css
Sao chép
Chỉnh sửa
Clothes-Classification-with-MLflow/
├── DL.py               # Mã nguồn huấn luyện mô hình CNN

├── predict.py          # Mã nguồn dự đoán với mô hình đã huấn luyện

├── test_images/        # Thư mục chứa hình ảnh thử nghiệm

├── README.md           # Tệp hướng dẫn sử dụng

└── .gitignore          # Tệp cấu hình Git

DL.py: Đây là tệp mã nguồn chính để huấn luyện mô hình phân loại hình ảnh quần áo. Nó bao gồm các bước tiền xử lý dữ liệu, xây dựng và huấn luyện mô hình CNN.

predict.py: Tệp này dùng để tải mô hình đã huấn luyện và thực hiện dự đoán trên các hình ảnh thử nghiệm từ thư mục test_images/.

test_images/: Thư mục chứa các hình ảnh quần áo để kiểm thử mô hình.

🧠 Mô Tả Mô Hình
Mạng Nơ-ron Tích Chập (CNN): Mô hình sử dụng các lớp Convolutional để trích xuất đặc trưng từ hình ảnh và các lớp Fully Connected để phân loại.

Thành phần chính:

Lớp Conv2D (Convolutional) giúp trích xuất các đặc trưng không gian của hình ảnh.

Lớp MaxPooling2D giúp giảm độ phân giải và giữ lại các đặc trưng quan trọng.

Lớp Fully Connected (Dense) cuối cùng giúp phân loại hình ảnh vào các nhãn quần áo.

Hàm kích hoạt: ReLU cho các lớp ẩn và Softmax cho lớp đầu ra để phân loại đa lớp.

Hàm mất mát: CrossEntropyLoss được sử dụng cho bài toán phân loại đa lớp.

Tối ưu hóa: Thuật toán Adam được sử dụng để tối ưu hóa hàm mất mát.

🚀 Hướng Dẫn Sử Dụng
1. Cài Đặt Các Thư Viện Cần Thiết
Trước khi chạy mã nguồn, bạn cần cài đặt các thư viện cần thiết. Dưới đây là cách cài đặt các thư viện trong Python:

bash
Sao chép
Chỉnh sửa
pip install torch torchvision mlflow matplotlib
PyTorch và Torchvision sẽ được sử dụng để xây dựng mô hình học sâu.

MLflow được sử dụng để theo dõi quá trình huấn luyện.

Matplotlib sẽ giúp vẽ các biểu đồ trực quan hóa trong quá trình huấn luyện.

2. Huấn Luyện Mô Hình
Để huấn luyện mô hình, bạn chỉ cần chạy tệp DL.py:

bash
Sao chép
Chỉnh sửa
python DL.py
Mô hình sẽ được huấn luyện trên một tập dữ liệu hình ảnh quần áo (ví dụ: dataset như FashionMNIST hoặc CIFAR-10).

MLflow sẽ theo dõi các tham số huấn luyện (epoch, batch size, learning rate) và các chỉ số như độ chính xác và hàm mất mát trong quá trình huấn luyện.

Sau khi huấn luyện hoàn tất, mô hình sẽ được lưu lại để sử dụng cho các bước tiếp theo.

3. Dự Đoán Với Hình Ảnh Mới
Sau khi huấn luyện mô hình, bạn có thể sử dụng mô hình đã huấn luyện để dự đoán với hình ảnh mới:

bash
Sao chép
Chỉnh sửa
python predict.py
Tệp predict.py sẽ tải mô hình đã huấn luyện và sử dụng các hình ảnh từ thư mục test_images/ để dự đoán loại quần áo.

Các kết quả dự đoán sẽ được hiển thị trên màn hình, hoặc có thể được lưu lại tùy theo cách triển khai trong mã nguồn.

4. Theo Dõi Quá Trình Huấn Luyện với MLflow
MLflow giúp bạn dễ dàng theo dõi các tham số và kết quả của các thí nghiệm. Để xem kết quả huấn luyện của mình, bạn có thể khởi chạy giao diện người dùng của MLflow:

bash
Sao chép
Chỉnh sửa
mlflow ui
Sau khi chạy lệnh trên, mở trình duyệt và truy cập http://localhost:5000 để xem các thí nghiệm đã lưu trữ, bao gồm độ chính xác, hàm mất mát, và các tham số mô hình.

📊 Kết Quả Mong Đợi
Độ chính xác: Mô hình phân loại hình ảnh quần áo có thể đạt độ chính xác cao tùy thuộc vào chất lượng dữ liệu huấn luyện và cấu trúc mô hình.

Biểu đồ: MLflow sẽ lưu các biểu đồ về loss và accuracy trong quá trình huấn luyện, giúp bạn đánh giá mô hình một cách trực quan.

