# RANSAC UI/UX refinement · 2026-10-10

Đây là đợt tinh chỉnh UI/UX của RANSAC MVP, không thay `utils/ransac.py`, deterministic fixtures, Chapter 1 hay Manim.

## Thay đổi

- RMS sau TLS refit dùng delta dương/màu xanh khi giảm so với candidate trên cùng winning consensus. Caption nói rõ RMS vuông góc càng thấp càng tốt.
- Guided trên tablet/mobile dùng tiến trình gọn thay lưới 10 ô, nút Previous/Next/Reset trên một hàng; biểu đồ chính được đưa lên trước thẻ giải thích.
- Presentation Mode tăng vùng vẽ, font/marker/legend và giảm phần UI phụ; cùng dữ liệu và công thức với chế độ thường.
- Experiment Lab xem từng trial thực qua Sample → Fit → Measure → Consensus → Update Best. Candidate, best consensus model và final TLS refit có màu/legend riêng; stage và score lấy từ `RansacResult`/`Trial` thực.
- Giải thích ngắn bằng tiếng Việt cho candidate tốt/xấu, nhiều outlier, ε quá nhỏ/quá lớn và N hữu hạn. Giữ thuật ngữ kỹ thuật tiếng Anh khi cần.

## Verification

- `python -m unittest discover -s tests`: **21/21 passed** (17 test gốc + 4 regression tests). Cảnh báo Streamlit `missing ScriptRunContext` trong bare-mode test không phải exception của app.
- Streamlit server thật: `/_stcore/health` trả `ok`.
- Edge + Playwright: desktop **1920×1080**, laptop **1366×768**, tablet **768×1024**, mobile **390×844**. Guided đi hết 10 bước, kiểm tra prediction/reveal gates, Previous/Reset; Lab đi hết 5 pha, batch, toàn bộ N, reset; các preset và trường hợp N=1; Presentation Mode cho Guided và Lab.
- Không có horizontal overflow, heading/button/metric clipping, Streamlit exception hay browser console/page error trong các lượt kiểm tra.
- Preset thực nghiệm: clean **33/37**, many outliers **24/60**, ε quá nhỏ **11/42**, ε quá lớn **36/42**. Các score này là kết quả của cấu hình deterministic, không phải số hiển thị giả.

## Review screenshots

- Guided ban đầu: [desktop](docs/screenshots/refinement/desktop-guided-initial.png), [tablet](docs/screenshots/refinement/tablet-guided-initial.png), [mobile](docs/screenshots/refinement/mobile-guided-initial.png).
- [Dải ε và khoảng cách vuông góc](docs/screenshots/refinement/guided-step-4-plot.png), [TLS refit/RMS](docs/screenshots/refinement/guided-step-8.png).
- Presentation Mode: [Guided](docs/screenshots/refinement/presentation-guided.png), [Lab](docs/screenshots/refinement/presentation-lab.png).
- Lab iteration: [Sample](docs/screenshots/refinement/lab-sample.png) → [Fit](docs/screenshots/refinement/lab-fit.png) → [Measure](docs/screenshots/refinement/lab-measure-plot.png) → [Consensus](docs/screenshots/refinement/lab-consensus.png) → [Update Best](docs/screenshots/refinement/lab-update-best-plot.png); [mobile Update Best](docs/screenshots/refinement/mobile-lab-update-best.png).

## Giới hạn còn lại

- App vẫn chỉ fit một đường thẳng 2D; chưa có chọn/kéo điểm trực tiếp hoặc nạp point cloud từ file.
- Score cao không bảo đảm model đúng khi ε quá rộng; N hữu hạn chỉ tăng xác suất có sample tốt, không bảo đảm thành công.
- Ở 390 px, các nhãn pha Lab chia thành hai hàng; không bị cắt, và nút chuyển pha vẫn ở một hàng.

## Cleanup và Git

- Đã dừng riêng server Streamlit dùng port 8514; health endpoint không còn truy cập được, không còn tiến trình Edge/Node/Python có command line của đợt kiểm tra. Giữ 13 ảnh review ở `docs/screenshots/refinement/`.
- Đã xóa `.ui-refine-tools/` cùng ba thư mục `__pycache__` trong Chapter 3 (chỉ là cache sinh ra), tổng dung lượng file **16,583,283 byte**; dung lượng trống ổ C tăng **17,002,496 byte** trong phép đo ngay trước/sau cleanup. Dữ liệu nguồn, test, tài liệu và các screenshot audit cũ được giữ nguyên.
- Không commit/push. Nhánh hiện tại `feature/chapter3-manim-animations`; toàn bộ `chapter3_webapp/` vẫn đang untracked từ trước đợt tinh chỉnh.
