# Chapter 2 · Interactive Feature Lab

Ứng dụng Streamlit gồm năm module: Harris & Shi-Tomasi, Gaussian Scale Space/LoG/DoG, SIFT, FAST/BRIEF/ORB-style và Feature Matching Challenge. Module cuối là bài ứng dụng tích hợp, không phải thuật toán thứ năm trong slide. Luồng học chính: dự đoán → tương tác → quan sát phép tính thật → giải thích → thử nghiệm.

## Chạy và kiểm thử

Từ repository root, dùng môi trường `.venv` chung và dependency trong `requirements.txt` ở root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run chapter2_webapp\app.py
```

```powershell
Push-Location chapter2_webapp
..\.venv\Scripts\python.exe -B -m unittest discover -s tests
Pop-Location
```

Không cần tải dataset. Các cảnh dạy học được sinh deterministic. Guided có tổng cộng 79 bước và các prediction/reveal gate; Experiment Lab cho phép thay tham số, xem kết quả tương ứng; Presentation Mode tăng khả năng đọc khi chiếu lớp. Xem [Teaching Guide](TEACHING_GUIDE.md) để chuẩn bị câu hỏi, lịch 45/90 phút và bài tập.

## Cấu trúc

```text
app.py                  Chọn module, bố cục và theme responsive
.streamlit/config.toml  Cấu hình khi khởi động từ thư mục Chapter 2
tutorials/              Guided, Lab, Presentation và session state từng module
utils/                  Năm engine số học và các hàm trực quan hóa
tests/                  Unit/AppTest và sáu script Edge/CDP browser audit
docs/screenshots/       Ảnh minh họa được tuyển chọn; không phải runtime assets
```

Năm engine là `utils/harris.py`, `scale_space.py`, `sift.py`, `orb.py`, `feature_challenge.py`. SIFT tái sử dụng scale-space; Challenge tái sử dụng detector/descriptor đã học. Các file `*_visualization.py` biến kết quả tính thật thành hình; `tutorials/` điều phối nội dung và state. Không có ảnh trong `docs/` được tải bởi app hay tests.

Kiểm tra browser đầy đủ, từ repository root sau khi chạy server ở `127.0.0.1:8522`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run chapter2_webapp\app.py --server.headless true --server.address 127.0.0.1 --server.port 8522
node chapter2_webapp/tests/final_chapter2_browser_audit.mjs
```

Script tích hợp đi qua cả năm module và các chế độ; các script `*_browser_audit.mjs` còn lại là kiểm tra sâu từng module. Chúng dùng Edge DevTools cục bộ, không yêu cầu thư viện JS mới và có thể tạo ảnh kiểm thử mới trong `docs/screenshots/`; chỉ đưa ảnh đã tuyển chọn vào Git.

## Quy ước và giới hạn

- Ảnh số học là grayscale float `[0,1]`; mảng dùng `[row,col]`, biểu đồ dùng `(x=col,y=row)` với y hướng xuống. Ngưỡng và kết quả phụ thuộc cảnh/seed hiện tại.
- Harris và Shi-Tomasi dùng cùng tensor. Harris `R=det(M)−k·trace(M)²`; NMS và ngưỡng áp trên response thật. Minh họa một scale, không bảo đảm góc ổn định dưới resize/quay hoặc ở vùng đệm warp.
- Scale Space dùng Gaussian rời rạc với `σ_global=σ_local·2^octave`; LoG/DoG giữ dấu, `DoG=L(kσ)−L(σ)`. DoG candidate strict 26-neighbor chưa phải SIFT keypoint đã lọc. So DoG với LoG là xấp xỉ, không phải đẳng thức pixel.
- SIFT thực hiện localization/filtering/orientation/descriptor 128D; một vị trí có thể tạo nhiều orientation record. Lowe ratio là quyết định descriptor, không chứng minh đúng hình học. Đây là bản sư phạm, không tương đương từng số với OpenCV/Lowe.
- ORB-style dùng FAST-12, centroid, BRIEF 256 bit deterministic và Hamming. Pattern không phải learned rBRIEF của OpenCV; nearest, cross-check và ratio là các rule khác nhau.
- Challenge dùng cùng cặp ảnh synthetic và affine ground truth để **hậu kiểm**, không chọn match. Harris có patch descriptor để làm baseline. So sánh mặc định có thể dùng các matching rule khác nhau; chọn cùng rule nếu muốn so policy chung. Precision và coverage có mẫu số riêng; mẫu số rỗng hiển thị N/A. `coverage` không phải detector recall. Thời gian đo là teaching-engine timing, không suy ra tốc độ production.

## Ảnh minh họa

| Module | Guided/khái niệm | Lab/Presentation |
| --- | --- | --- |
| Harris | [NMS 5×5](docs/screenshots/final_audit/harris-nms-neighborhood.png), [vùng hỗ trợ warp](docs/screenshots/final_audit/harris-scale-support.png) | [Lab](docs/screenshots/final_audit/harris-lab-1.png), [Presentation](docs/screenshots/final_audit/harris-presentation-key-laptop.png) |
| Scale Space | [residual](docs/screenshots/final_audit/scale-step-4-before.png), [extrema](docs/screenshots/final_audit/scale-step-13-before.png) | [Lab](docs/screenshots/final_audit/scale-lab-2.png), [Presentation](docs/screenshots/final_audit/scale-presentation-key-laptop.png) |
| SIFT | [descriptor](docs/screenshots/final_audit/sift-step-13-before.png) | [Lab](docs/screenshots/final_audit/sift-lab-2.png), [Presentation](docs/screenshots/final_audit/sift-presentation-key-laptop.png) |
| ORB-style | [trước Reveal](docs/screenshots/final_audit/orb-step-17-before.png), [sau Reveal](docs/screenshots/final_audit/orb-step-17-after.png) | [Lab](docs/screenshots/final_audit/orb-lab-2.png), [Presentation](docs/screenshots/final_audit/orb-presentation-key-laptop.png) |
| Challenge | [trước hậu kiểm](docs/screenshots/final_audit/feature-step-10-before.png), [sau hậu kiểm](docs/screenshots/final_audit/feature-step-10-after.png) | [mobile cards](docs/screenshots/final_audit/feature-comparison-mobile-cards.png), [Presentation](docs/screenshots/final_audit/feature-presentation-key-laptop.png) |

Tổng quan: [desktop](docs/screenshots/final_audit/selector-desktop.png) · [mobile](docs/screenshots/final_audit/selector-mobile.png).
