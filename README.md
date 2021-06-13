<h1 align="center"> ĐỒ ÁN TỐT NGHIỆP ĐỢT GIAO THÁNG 03/2021 </h1>


## ĐỀ TÀI: NGHIÊN CỨU TRIỂN KHAI THUẬT TOÁN YOLOV3 VÀ XÂY DỰNG HỆ THỐNG HƯỚNG DẪN ĐẬU XE Ô TÔ PARKING VISION
### Giảng viên hướng dẫn: ThS. PHẠM THỊ MIÊN  
### Sinh viên thực hiện: NGUYỄN VĂN HIỆP
### Lớp: CÔNG NGHỆ THÔNG TIN
### Khoá: 58
#### 1. Mục đích, yêu cầu:
##### a. Mục đích
-	Xây dựng website theo dõi các vị trí còn trống và đã đỗ trong bãi đỗ xe và hiển thị lên màn hình số vị trí còn trống, đã đỗ và tổng số lượng chỗ đỗ trong bãi đỗ xe ô tô theo thời gian thực.
-	Xây dựng mô hình nhận diện các chỗ còn trống và đã đỗ trong bãi đỗ xe bằng hình ảnh, video, camera.
-	Xây dựng website giới thiệu sản phẩm.
-	Xây dựng giải pháp cho Hệ Thống Hướng Dẫn Bãi Đỗ Xe (Parking Guidance System) thông minh tiết kiệm, nhanh hơn, tiện dụng, chính xác và có thể mở rộng quy mô sau này.
##### b. Yêu cầu
-	Tìm hiểu về Thị giác Máy tính (Computer Vision) và Học Sâu (Deep Learning).
-	Nghiên cứu về xử lý ảnh.
-	Nghiên cứu những quy trình trong xử lý ảnh.
-	Nghiên cứu về Mạng Thần Kinh Tích Chập (Convolution Neural Network) và những ứng dụng của nó trong Deep Learning và Computer Vision.
-	Nghiên cứu thuật toán YOLO (You Only Look Once).
-	Tìm hiểu các nguồn cơ sở dữ liệu hình ảnh cho quá trình huấn luyện mô hình Học Sâu.
-	Tìm hiểu một số khái niệm liên quan đến lĩnh lực Khai Phá Dữ Liệu (Data Mining) áp dụng vào mô hình học sâu.
-	Thu thập dữ liệu hình ảnh về những chỗ trống và đã đỗ trong bãi đỗ xe. Gắn nhãn, tiền xử lý.
-	Tìm hiểu về Transfer Learning, và ứng dụng vào huấn luyện mô hình.
-	Ứng dụng kiến trúc Darknet huấn luyện mô hình trên tập dữ liệu lớn và tập dữ liệu mô phỏng bằng công cụ Google Colab.
-	Ứng dụng thuật toán YOLOv3 (You Only Look Once, Version 3) để phát hiện những vị trí còn trống, đã đỗ trên hình ảnh, video, camera bằng Pytorch.
-	Ứng dụng Django để xây dựng website giới thiệu sản phẩm, truyền màn hình đã xử lý lên website.
#### 2. Nội dung và phạm vi đề tài:
##### a. Nội dung đề tài
-	Giới thiệu và phân biệt các khái niệm liên quan đến Trí tuệ Nhân tạo (Artificial Intelligence), Thị giác Máy tính (Computer Vision), Học Máy (Machine Learning), Học Sâu (Deep Learning).
- Nghiên cứu và triển khai thuật toán YOLOv3 bằng Pytorch:
  - Triển khai module nhận diện qua hình ảnh
  - triển khai module nhận diện qua video/camera
-	Kiểm thử mô hình.
-	Nghiên cứu các chỉ số đánh giá mô hình.
-	Xây dựng trang web hiển thị màn hình xử lý bằng Django.
-	Hiển thị song song màn hình xử lý trên website và trên desktop.
##### b. Phạm vi đề tài.
-	Bài toán nhận diện vật thể (Object Detection).
-	Giới thiệu họ các thuật toán Mạng Thần kinh Tích chập (Convolution Neural Network) trong nhận diện vật thể.
-	Ứng dụng Django để xây dựng website giới thiệu sản phẩm.
-	Tích hợp module nhận diện qua camera vào Django để truyền màn hình đã xử lý lên website.

#### 3. Công nghệ, công cụ và ngôn ngữ lập trình:
##### Công nghệ: Python, OpenCV, Pytorch, Django, Colab Notebook, CUDA, Darknet
##### Công cụ:
- Một số thư viện mã nguồn mở của Python: opencv-python, pandas, numpy, django, torch,...
- Visual Studio Code
- Darknet: Open Source Neural Networks
- Google Colab
##### Ngôn ngữ lập trình: Python
#### 4. Các kết quả chính dự kiến sẽ đạt được và ứng dụng:
-	Sử dụng camera tiến hành phát hiện các vị trí còn trống và đã đỗ trong thời gian thực. Hiển thị lên màn hình vị trí còn trống và đã đỗ, đếm những vị trí còn trống, đã đỗ, tổng các vị trí hiện có.
-	Sử dụng Django để xây dựng website.
-	Mô phỏng mô hình trực tiếp.
-	Hoàn chỉnh cuốn báo cáo đề tài.
-	Nắm được kiến trúc thuật toán YOLOv3 và có thể ứng dụng vào mọi đề tài liên quan.
-	Nắm được các ưu, nhược điểm của thuật toán và các phương pháp tối ưu cho thuật toán.
-	Nắm được những quy trình trong huấn luyện và kiểm tra mô hình trong các mô hình Deep Learning.
