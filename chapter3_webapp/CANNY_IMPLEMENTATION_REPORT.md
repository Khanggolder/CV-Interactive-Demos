# Canny Edge Detector · implementation & verification

## Scope và kiến trúc

Module thứ tư trong `chapter3_webapp/`, không đổi Chapter 1, approved Manim scenes, RANSAC/Hough/Sobel numerical engines, requirements hay frontend framework. Không commit/push/merge.

| File | Vai trò |
| --- | --- |
| `app.py` | Module selector, mode, Presentation và CSS responsive Canny |
| `utils/canny.py` | NumPy-only image presets, Gaussian, signed Sobel, normalization, NMS, threshold, BFS event replay |
| `utils/canny_visualization.py` | Plotly stage/pixel/patch/kernal views từ arrays thật, không quyết định thuật toán |
| `tutorials/canny.py` | Guided 14 bước, Lab 2 workspace, cache, `canny_*` state, prediction/reveal, BFS controls |
| `tests/test_canny.py`, `tests/test_canny_app.py` | 22 numerical + 7 integration regression tests |
| `README.md`, `CANNY_IMPLEMENTATION_REPORT.md` | Hướng dẫn, conventions, kiểm chứng |
| `docs/screenshots/canny/` | 19 ảnh Edge review ở desktop/laptop/tablet/mobile |

## Numerical pipeline

Mặc định: ảnh synthetic **224×224, seed 13**, Gaussian **5×5**, **σ=1.15**, low **0.10**, high **0.35**. Generator giữ đúng circle, rectangle, diagonal, ring, background, noise, texture và uint8 conversion của `animations/chapter3/canny_pipeline.py`. Original được đổi sang `[0,1]`. Gaussian weights từ công thức và chuẩn hóa tổng 1; border `reflect`. Kx/Ky của Sobel được áp theo **cross-correlation** (không flip); rows/y ảnh tăng xuống. Gx/Gy giữ dấu, `M_raw=hypot(Gx,Gy)`, `θ=atan2(Gy,Gx)`; M=0 hiển thị hướng Undefined. `M_norm=clip(M_raw/max(P99.5,1e−9),0,1)`.

NMS dùng `M_norm` và 4 sector 0°/45°/90°/135°, q1/p/q2 là đúng tọa độ ảnh. KEEP khi p ≥ cả hai neighbor, kể cả tie; border một pixel =0. NMS output được chuẩn hóa **lần hai** bởi max trước double threshold. Strong ≥high, weak low≤value<high, zero background bị loại. BFS 8-connected bắt đầu từ mọi strong; accepted weak vẫn là weak. Replay lưu thứ tự pop/discovery thật; UI nhảy qua những pop không thêm weak, caption ghi số event gốc. Run to completion xét hết queue, bằng final binary map. Không dùng `cv2.Canny()` để thay pipeline.

Đối chiếu trực tiếp engine với approved Manim scene trên ảnh reference mặc định: ảnh gốc và cả 8 mảng `original`, `smoothed`, `magnitude`, `direction`, `nms`, `strong`, `weak`, `edges` **array_equal** (sai khác lớn nhất 0). Default có **1500 strong, 447 weak, 155 accepted weak, 292 isolated weak**, **1655** BFS pop event. Reference NMS `(130,116)` KEEP, `(130,115)` SUPPRESS. Preset D có **332 strong, 115 weak, 115 connected weak**; preset E có **176 strong, 136 weak, 136 isolated weak** ở tham số mặc định. Các con số này là kết quả engine, không hardcode vào UI.

## Trải nghiệm học và state

Guided 14 bước: vấn đề biên/noise → overview → Gaussian kernel/smoothing → Gx/Gy → magnitude/direction → ridge dày → NMS KEEP và SUPPRESS → NMS toàn ảnh → double threshold → BFS connected weak → isolated weak → final/limitations. NMS và threshold có prediction/reveal gate; Previous/Next/Reset giữ đúng tiến trình. Inspector 7×7, các marker p/q1/q2 và giá trị so sánh luôn dùng cùng mảng với quyết định NMS. Lab A cho chọn 9 stage, preset, σ, low/high, pixel và so sánh ảnh gốc. Lab B cho NMS microscope và BFS với tiến/lùi/chạy 20 discovery/hoàn tất/reset. Presentation tăng kích thước chart/nhãn, giảm clutter. Mobile stack chart và có progress gọn; BFS controls xuống 2 cột.

Cache base theo image bytes/shape/σ, threshold theo thêm low/high. Đổi ảnh/σ/threshold reset replay và reveal; chọn stage/pixel không tính lại pipeline; Streamlit rerun không tự tăng BFS. Invalid `0<low<high≤1` báo lỗi, không tráo ngưỡng. Preset được tái tạo deterministic; ảnh reference luôn có thể chọn lại.

## Tests và browser audit

`python -B -m unittest discover -s tests` từ `chapter3_webapp/` bằng Python có `requirements.txt`: **93/93 passed** (64 baseline + 29 Canny), 0 failed. Bài test gồm Gaussian kernel/border, signed Sobel, raw/normalized magnitude, 4 NMS sectors và góc biên, tie/border, threshold/zero exclusion, 8-connectivity/multihop/isolated, deterministic replay, reference counts, preset outcome, parameter invalidation, Guided gates, Lab controls, pixel sync, Presentation và module regression. Warning `missing ScriptRunContext` trong bare-mode test không phải Streamlit app exception. Tệp mới được import khi chạy full suite; kiểm tra độc lập trailing whitespace không phát hiện lỗi; `git diff --check` chỉ có ý nghĩa hạn chế vì Chapter 3 đang untracked.

Streamlit server thật port **8524** trả `/_stcore/health` → `ok`. Edge headless + Playwright: desktop **1920×1080**, laptop **1366×768**, tablet **768×1024**, mobile **390×844**. Desktop đi hết 14 Guided steps, prediction gates, NMS, threshold, BFS partial/final, Lab/Presentation, đổi lại Sobel/Hough/RANSAC. Browser Lab còn chọn preset connected weak, đổi σ/low/high, thử invalid threshold (cảnh báo xuất hiện), chọn NMS shoulder, chạy/hoàn tất/reset BFS. Mobile đã kiểm tra vùng inspector và BFS controls sau khi scroll. Không có `stException`, page/console error hoặc horizontal overflow ở lượt audit cuối. Screenshot được mở xem trực tiếp; ảnh browser chứng minh bố cục/trạng thái hiển thị, phép toán được xác nhận bằng tests và so sánh Manim, không suy ra độ đúng số học chỉ từ ảnh.

Ảnh đại diện: [Guided](docs/screenshots/canny/desktop-guided-initial.png), [Gaussian](docs/screenshots/canny/desktop-gaussian.png), [gradient](docs/screenshots/canny/desktop-gradient.png), [NMS KEEP](docs/screenshots/canny/desktop-nms-keep.png), [NMS SUPPRESS](docs/screenshots/canny/desktop-nms-suppress.png), [threshold](docs/screenshots/canny/desktop-threshold.png), [BFS partial](docs/screenshots/canny/desktop-hysteresis-partial.png), [BFS final](docs/screenshots/canny/desktop-hysteresis-complete.png), [isolated weak](docs/screenshots/canny/desktop-isolated-weak.png), [final map](docs/screenshots/canny/desktop-final.png), [Lab](docs/screenshots/canny/desktop-lab.png), [Presentation](docs/screenshots/canny/desktop-presentation-bfs.png), [mobile Guided](docs/screenshots/canny/mobile-guided.png), [mobile NMS inspector](docs/screenshots/canny/mobile-nms-inspector.png), [mobile BFS](docs/screenshots/canny/mobile-hysteresis-controls.png), [mobile Lab](docs/screenshots/canny/mobile-lab-image.png).

## Lỗi phát hiện và sửa

- Preset connected weak ban đầu có khoảng trống sau NMS nên weak không nối được; thay chuyển tiếp intensity deterministic và xác nhận 115 accepted weak thật.
- Replay “Next” ban đầu phải đợi hàng trăm strong pop mới thấy weak đầu tiên; UI giờ nhảy giữa *event thật có discovery*, không tạo frame giả; hiện số pop gốc.
- BFS Start/Reset ban đầu không đưa step về 0; integration test bắt và đã sửa.
- Plotly `scaleanchor` ở mobile từng mở rộng trục đến row âm và để vùng trống lớn; ràng buộc domain để giữ pixel vuông và tọa độ thật.
- NMS marker text/queue label có độ tương phản và chồng chữ; tăng tương phản, giữ queue bằng viền marker không lặp nhãn.
- Lab và Presentation trước đó đẩy hình chính xuống sâu; tham số/prediction chuyển vào expander, vẫn truy cập được khi cần.

## Giới hạn

Đây là NMS lượng tử 4 hướng, không phải interpolated subpixel NMS; plateau tie có thể giữ nhiều pixel. Các threshold chuẩn hóa theo từng ảnh không phải một mức intensity vật lý cố định. 8-connectivity có thể nối biên gần nhau ngoài ý muốn; blur mạnh làm mất nét mảnh. Preset clean/fine không nhất thiết có weak pixels ở mọi tham số; UI hiển thị tình trạng thật và không giả lập nhãn. Streamlit cập nhật theo từng tương tác, không chạy video/drag trực tiếp. Screenshot Streamlit là viewport của vùng cuộn; những ảnh mobile ở phần dưới được chụp sau khi cuộn, không phải toàn trang ghép giả.

## Cleanup và Git

Đã dừng riêng Streamlit server port 8524; health endpoint không còn phục vụ. Mọi Edge Playwright context/browser từ tác vụ đã `close()` và Node audit process đã kết thúc. Đã xóa riêng `chapter3_webapp/.canny-audit-tools/` (Playwright Core và script tạm, **8,242,851 file bytes**) cùng 7 screenshot thử nghiệm/trùng (**983,617 bytes**); tổng **9,226,468 file bytes**. Dung lượng trống C: **187,035,979,776 → 187,046,035,456 bytes** (tăng 10,055,680 bytes theo ổ đĩa). Giữ **19** screenshot review, tất cả source/tests/docs/config, Chapter 1, các module trước, Manim, video và cache dùng chung.

`git status --short --branch`: branch `feature/chapter3-manim-animations`, mục `?? chapter3_webapp/` vì toàn bộ thư mục đã untracked từ trước. Không stage/commit/push/merge/rebase.
