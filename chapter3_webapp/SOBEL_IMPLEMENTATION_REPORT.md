# Sobel Convolution · implementation & verification

## Scope và kiến trúc

- `app.py`: thêm module thứ ba, giữ navigation/Presentation Mode hiện có; không sửa thuật toán RANSAC/Hough.
- `utils/sobel.py`: engine NumPy thuần; validate ảnh 2D integer 0–255, patch 3×3 interior, products/row sums, signed Gx/Gy, M/θ, full gradient maps, edit pixel, sáu preset fixed-seed. Không import Streamlit/Plotly/Manim.
- `utils/sobel_visualization.py`: matrix/gradient/vector Plotly lấy số trực tiếp từ engine; màu display tách khỏi số thô.
- `tutorials/sobel.py`: Guided 12 bước, prediction/reveal, stepwise multiplication, hai Lab workspace, `sobel_*` session state, reset/invalidate theo input.
- `tests/test_sobel.py`, `tests/test_sobel_app.py`: số học độc lập, biên, overflow, normalization, interaction và module regression.
- `README.md`: cập nhật cấu trúc, lệnh chạy, quy ước và giới hạn.
- `docs/screenshots/sobel/`: ảnh review Edge ở bốn viewport.

Không thay Chapter 1, scene Manim hay requirements; không thêm dependency runtime. Không commit/push/merge.

## Quy ước và khả năng tương tác

Mask được áp **cross-correlation** trực tiếp như scene duyệt, không lật kernel. Rows/y ảnh tăng xuống, cols/x tăng sang phải. Gx/Gy là số nguyên có dấu; M=`hypot(Gx,Gy)`; θ=`atan2(Gy,Gx)` theo tọa độ ảnh. Nếu M=0, θ hiển thị **Undefined**. Vector Plotly y hướng lên dùng `(Gx,-Gy)`. Chỉ valid interior center; border map NaN/`—`, không padding. Input image dùng thang 0–255 cố định; Gx/Gy dùng diverging color đối xứng, magnitude chuẩn hóa để tô màu, direction dùng bảng màu tuần hoàn. Nhãn ô luôn là raw value.

Guided đi từ pixel, patch, Gx mask, chín tích, tổng Gx, Gy cùng patch, vector, M/θ, full map, sáu preset, clean/noisy comparison đến concept check Canny. Gy và kết luận yêu cầu predict trước reveal. Lab A cho inspect/edit pixel, di chuyển center từng ô, Gx/Gy và điều khiển từng product; Lab B cho bốn map, ô đang chọn và vector. Thay ảnh/center/mask sẽ xóa tiến độ tính cũ; chọn pixel cập nhật giá trị edit đang đề xuất. Presentation Mode tăng kích thước hình và ma trận, không thay toán học.

## Giá trị từ preset ở center (2,3)

| Preset | Gx | Gy | M | θ (image coords) |
| --- | ---: | ---: | ---: | --- |
| A · Vertical edge | 720 | 0 | 720.000 | 0° |
| B · Horizontal edge | 0 | 720 | 720.000 | 90° |
| C · Diagonal edge | 540 | 540 | 763.675 | 45° |
| D · Flat region | 0 | 0 | 0.000 | Undefined |
| E · Reversed contrast | −720 | 0 | 720.000 | 180° |
| F · Noisy edge (seed 37) | 742 | 26 | 742.455 | 2.007° |

Những số này được tính bằng engine cho cùng center, không hardcode vào UI. Test đối chiếu per-pixel với công thức weighted left/right và top/bottom độc lập. Với patch saturated `uint8` `[0,0,255]×3`, Gx=1020, không overflow.

## Tests và browser audit

`python -B -m unittest discover -s tests` (từ `chapter3_webapp/`, bằng Python environment có requirements): **64/64 passed**, gồm 39 test RANSAC/Hough trước đó và 25 test Sobel mới. Bare-mode Streamlit có warning `missing ScriptRunContext`; test không có app exception sau sửa.

Streamlit server thật port 8523: `/_stcore/health` → `ok`. Edge headless + Playwright: desktop 1920×1080, laptop 1366×768, tablet 768×1024, mobile 390×844. Browser audit đi hết Guided 12 bước trên desktop; thao tác prediction/reveal, product controls, sửa pixel, đổi preset (browser xác nhận Gy=720 cho horizontal, Gx=Gy=540 cho diagonal, θ Undefined cho flat), gradient map, Presentation Mode, mobile product/map và chuyển lại RANSAC/Hough. Không có browser console/page error, Streamlit exception hay horizontal overflow ở lượt cuối. Ảnh chỉ chứng minh bố cục/trạng thái hiển thị; kết quả số học do test NumPy xác nhận.

Ảnh tiêu biểu: [Guided ban đầu](docs/screenshots/sobel/desktop-guided-initial.png), [tính tích](docs/screenshots/sobel/desktop-product.png), [Gx](docs/screenshots/sobel/desktop-gx.png), [Gy](docs/screenshots/sobel/desktop-gy.png), [vector](docs/screenshots/sobel/desktop-vector.png), [diagonal map](docs/screenshots/sobel/desktop-diagonal-map-detail.png), [flat/Undefined](docs/screenshots/sobel/desktop-flat.png), [Presentation Mode](docs/screenshots/sobel/desktop-presentation.png), [tablet](docs/screenshots/sobel/tablet-guided-map.png), [mobile Guided](docs/screenshots/sobel/mobile-guided-initial.png), [mobile Lab](docs/screenshots/sobel/mobile-product-controls.png).

## Lỗi tìm thấy và sửa

- Map layer ban đầu có thể là `None` trong Streamlit test: đặt default rõ ràng.
- Plotly equal-scale ban đầu kéo trục x ra ngoài ảnh nhỏ: cố định đúng 0–6 và 0–4, tăng chiều cao và giới hạn chiều rộng chart.
- Chart phụ chạm nhãn trục: bỏ note trong plot, giữ giải thích dưới chart.
- Trùng Streamlit container key khi Lab B vừa có input image vừa có map: dùng key riêng.
- Vùng phẳng intensity 100 bị chuẩn hóa thành màu quá sáng, nhãn trắng khó đọc: input image giờ dùng thang cố định 0–255.
- Chọn pixel khác từng giữ đề xuất intensity cũ: đồng bộ trường edit theo pixel được chọn; callback preset chịu được widget selection bị Streamlit dọn khi không hiển thị.

## Giới hạn

Plotly/Streamlit cập nhật theo từng hành động, không phải video 60 fps hoặc drag/drop. Ảnh minh họa 5×7 deterministic; không xử lý file ảnh tùy ý. Noise/texture/corner có thể cho gradient không tương ứng biên vật thể; chưa có smoothing, NMS hay hysteresis. Chưa triển khai Canny. Full map chỉ vùng interior hợp lệ.

## Cleanup và Git

Đã dừng đúng Streamlit server port 8523; port không còn listener. Mọi Edge session được `browser.close()` trong Playwright và tiến trình test Node đã kết thúc. Xóa riêng `chapter3_webapp/.sobel-audit-tools/` (Playwright Core cài tạm, **13,665,614 byte**) và 9 ảnh QA trùng/dư trong `docs/screenshots/sobel/` (**802,465 byte**), giữ **17** ảnh review. Tổng file bytes đã xóa **14,468,079**; free space ổ C tăng **14,782,464 byte** theo đo ngay trước/sau cleanup. Không xóa source, test, docs, config, ảnh RANSAC/Hough, Chapter 1 hay Manim. `git status --short` cuối: `?? chapter3_webapp/` vì toàn bộ Chapter 3 webapp đã untracked từ trước; branch `feature/chapter3-manim-animations`. Không commit/push.
