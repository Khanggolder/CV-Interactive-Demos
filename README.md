# CV Interactive Demos

Phòng học tương tác cho môn **Xử lý ảnh và Thị giác máy tính**. Người học dự đoán, thay đổi tham số, quan sát các bước tính thật và giải thích giới hạn của thuật toán.

**Một root `.venv` · ba Streamlit apps · bảy Manim scenes đã duyệt.** Python 3.11 là phiên bản đã kiểm chứng trên Windows. Stack chính: Streamlit, NumPy, OpenCV headless, Pillow, Plotly và Manim 0.19.0. Không cần tải dataset lớn; sample giảng dạy có sẵn hoặc được sinh deterministic.

## Chạy từ repository root (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run chapter1_webapp\app.py
```

Mở terminal khác để chạy Chapter 2 hoặc Chapter 3:

```powershell
.\.venv\Scripts\python.exe -m streamlit run chapter2_webapp\app.py
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run chapter3_webapp\app.py
```

Không tạo `.venv` riêng trong từng chapter. Dừng server bằng `Ctrl+C`. Để chạy test với đúng import path, vẫn dùng **root interpreter** nhưng chuyển working directory:

```powershell
Push-Location chapter1_webapp
..\.venv\Scripts\python.exe -B -m unittest discover -s tests
Pop-Location
Push-Location chapter2_webapp
..\.venv\Scripts\python.exe -B -m unittest discover -s tests
Pop-Location
Push-Location chapter3_webapp
..\.venv\Scripts\python.exe -B -m unittest discover -s tests
Pop-Location
```

Manim verifier chạy từ `animations/chapter3/` bằng `..\..\.venv\Scripts\python.exe`, ví dụ `verify_lsd.py` hoặc `verify_sobel.py --preflight`. Xem [hướng dẫn Manim](animations/chapter3/README.md) để render/kiểm tra scene. FFmpeg CLI có thể cần cho workflow render/export cụ thể; các numerical verifier và video decode dùng PyAV của Manim.

## Nội dung học

| Thành phần | Tương tác chính |
| --- | --- |
| [Chapter 1](chapter1_webapp/README.md) | Pixel Explorer; Brightness/Contrast; Gamma; Histogram/Equalization/CLAHE; Noise và Gaussian/Median; Geometry/inverse mapping/interpolation; Fourier Ideal/Gaussian LP; Convolution kernels. |
| [Chapter 2](chapter2_webapp/README.md) | Harris & Shi-Tomasi (14 bước); Gaussian Scale Space, LoG & DoG (15 bước); SIFT (18 bước); FAST/BRIEF/ORB-style (18 bước); Feature Matching Challenge (14 bước, ba Lab và Classroom Challenge) tích hợp ba detector/descriptor trên cùng cặp ảnh. |
| [Chapter 3](chapter3_webapp/README.md) | RANSAC; Hough Line + Accumulator; Sobel; Canny; Hough Circle; LSD. Mỗi module có Guided Tutorial, Experiment Lab, Presentation Mode và visual tính từ state thật. |
| [Manim Chapter 3](animations/chapter3/README.md) | Bảy scene được duyệt: Canny, Hough point-to-curve, Hough accumulator, RANSAC, LSD, Sobel, Hough Circle. |

Luồng dạy học chủ đạo là **Predict → Interact → Observe → Explain → Experiment**. Chế độ Guided giữ các prediction/reveal gate; Lab cho phép thử dữ liệu và tham số; Presentation Mode tăng độ đọc khi trình chiếu.

Chapter 2 có [Teaching Guide](chapter2_webapp/TEACHING_GUIDE.md) để chuẩn bị buổi học; hướng dẫn chạy, giới hạn và ảnh minh họa nằm trong [README của Chapter 2](chapter2_webapp/README.md).

## Cấu trúc

```text
requirements.txt              # bộ dependency canonical cho Python 3.11
chapter1_webapp/              # app, utils, tests, sample images
chapter2_webapp/              # Harris/Shi-Tomasi, Scale Space/LoG/DoG, SIFT, ORB-style, Feature Matching Challenge
chapter3_webapp/              # app, sáu tutorial, numerical engines, tests
animations/chapter3/          # bảy Manim scenes, verifier; media cục bộ Git-ignored
```

## Giới hạn và trạng thái

Đây là **teaching demos**, không phải detector production. RANSAC hữu hạn lần thử không bảo đảm nghiệm tối ưu; Hough dùng vote/bin rời rạc; Sobel giữ gradient có dấu; Canny NMS lượng tử bốn hướng; Hough Circle trả bin center; LSD dùng teaching-model NFA, không phải full reference LSD. Kết quả cụ thể phụ thuộc fixture, ngưỡng và tương tác. Chapter 2 có bốn bài thuật toán và một Feature Matching Challenge ứng dụng tổng hợp; Chapter 4–5 vẫn là roadmap. Quy ước và giới hạn chi tiết nằm trong README của từng chapter.
