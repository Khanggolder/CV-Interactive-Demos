# Chapter 1 Interactive CV Demo

Ứng dụng Streamlit phục vụ giảng dạy trực tiếp Chương 1: từ pixel, toán tử điểm và histogram đến nhiễu, biến đổi hình học, Fourier và kernel tích chập.

## Setup on Windows · one root venv

Tạo môi trường chung từ repository root bằng Python 3.11. `requirements.txt` ở root là manifest canonical cho cả ba Streamlit apps và Manim; không tạo venv con trong chapter.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Push-Location chapter1_webapp
..\.venv\Scripts\python.exe -m streamlit run app.py
Pop-Location
```

Kiểm tra: `Push-Location chapter1_webapp; ..\.venv\Scripts\python.exe -B -m unittest discover -s tests; Pop-Location`.

## Stop

Nhấn `Ctrl+C` để dừng Streamlit; `Pop-Location` chạy sau khi server dừng. Working directory của chapter giúp nạp theme trong `.streamlit/config.toml`; không cần activate/deactivate.

## Kiểm tra trên Edge thật

Chạy server từ thư mục Chapter 1 với `--server.headless true --server.address 127.0.0.1 --server.port 8531`. Trong terminal khác ở repository root, chạy `node scripts/browser_audit.mjs chapter1`. Script kiểm tra màn hình đầu ở bốn viewport; sau đó đi qua tám trang và một số controls/upload/reveal ở laptop. Ảnh QA vào `.audit_output/chapter1/` (Git-ignored). Không cần package npm mới. Dừng server trước khi `Pop-Location`.

Đặt `CV_AUDIT_MOBILE_ONLY=1` nếu chỉ muốn kiểm tra thêm Geometry ở tablet và Gamma/Fourier ở mobile.

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
