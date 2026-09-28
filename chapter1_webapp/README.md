# Chapter 1 Interactive CV Demo

Ứng dụng Streamlit phục vụ giảng dạy trực tiếp Chương 1: từ pixel, toán tử điểm và histogram đến biến đổi Fourier và thư viện kernel tích chập.

## Setup on Windows

```powershell
cd chapter1_webapp
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Nếu lệnh `py` không hoạt động, dùng `python -m venv .venv`.

## Stop / deactivate

Nhấn `Ctrl+C` để dừng Streamlit, sau đó:

```powershell
deactivate
```

## Structure

- `app.py`: giao diện, điều hướng và luồng giảng dạy.
- `utils/image_ops.py`: toán tử điểm, cân bằng histogram và CLAHE.
- `utils/fourier.py`: FFT, mask tần số và tái tạo ảnh.
- `utils/visualization.py`: biểu đồ histogram và gamma bằng Plotly.
- `assets/samples/`: sáu ảnh mẫu nhỏ cho từng mục đích demo.
- `scripts/generate_samples.py`: tái tạo bộ ảnh mẫu mà không tải dataset.
- `tests/test_ops.py`: kiểm tra xử lý ảnh grayscale/RGB và Fourier.
- `tests/test_app.py`: render và tương tác qua toàn bộ sáu trang Streamlit.

## Teaching flow

Pixel → Brightness/Contrast → Gamma → Histogram → Fourier → Convolution Demo

Ở mỗi phần, dừng tại bước trung gian và yêu cầu sinh viên dự đoán kết quả trước khi tiếp tục.
Trong trang Fourier, kết quả lọc được ẩn mặc định cho tới khi chọn **Đã dự đoán**.
