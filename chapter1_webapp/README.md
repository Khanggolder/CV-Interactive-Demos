# Chapter 1 Interactive CV Demo

Ứng dụng Streamlit phục vụ giảng dạy trực tiếp Chương 1: từ pixel, toán tử điểm và histogram đến nhiễu, biến đổi hình học, Fourier và kernel tích chập.

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
- `utils/noise.py`: Gaussian/salt-and-pepper noise và bộ lọc Gaussian/Median.
- `utils/geometry.py`: ma trận đồng nhất, inverse mapping và nội suy.
- `utils/fourier.py`: FFT, Ideal/Gaussian mask và tái tạo ảnh.
- `utils/visualization.py`: biểu đồ histogram, gamma và intensity profile.
- `assets/samples/`: ảnh mẫu nhỏ, gồm hai teaching pattern tổng hợp.
- `scripts/generate_samples.py`: tái tạo bộ ảnh mẫu mà không tải dataset.
- `tests/test_ops.py`: kiểm tra toán tử ảnh, noise, geometry và Fourier.
- `tests/test_app.py`: render và tương tác qua toàn bộ tám trang Streamlit.

## Teaching flow

Pixel → Brightness/Contrast → Gamma → Histogram → Noise & Filtering → Geometric Transform → Fourier → Convolution Kernel Reference

Ở mỗi phần, dừng tại bước trung gian và yêu cầu sinh viên dự đoán kết quả trước khi tiếp tục.
Kết quả Noise, Geometry và Fourier được ẩn mặc định cho tới khi chọn **Đã dự đoán**.
