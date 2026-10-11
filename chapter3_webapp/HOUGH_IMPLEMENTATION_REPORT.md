# Hough Line + Accumulator · Implementation report

## Scope và kiến trúc

- `app.py`: module selector RANSAC/Hough, giữ nguyên RANSAC Guided/Lab và session state của nó.
- `utils/hough.py`: geometry normal-form, signed ρ, half-open bins, pure batch accumulator, immutable stepwise VoteState. Không import Streamlit/Manim.
- `utils/hough_visualization.py`: hai Plotly panel đồng bộ cho Image Space, continuous Hough Space và heatmap Accumulator.
- `tutorials/hough.py`: Guided 12 bước, prediction/reveal, Lab 2 workspace, presets và trạng thái `hough_*` riêng.
- `tests/test_hough.py`, `tests/test_hough_app.py`: số học, đồng bộ hình/marker và tương tác Streamlit.
- `README.md`: hướng dẫn chạy, conventions, fixture và giới hạn.

Không thay `utils/ransac.py`, Chapter 1 hay các scene Manim đã duyệt. Không thêm dependency.

## Guided và Lab

Guided đi từ edge points đến normal vector và signed ρ, point-to-sinusoid, giao điểm hai đường cong (prediction trước reveal), thêm bốn point thẳng hàng, binning, vote từng θ sample, peak và line lấy từ **bin center**. Bước cuối cho thử θ/ρ thô, ít support, clutter và threshold. Previous/Next/Reset giữ nhất quán; tablet/mobile dùng progress gọn.

Lab dùng cùng point dataset cho Point-to-Sinusoid và Accumulator Voting. Có active point, θ slider, custom point, bốn preset, Δθ/Δρ, threshold, Next Vote, Vote Selected, Vote All và reset. Bất kỳ thay đổi points/bin width đều reset vote state; đổi θ đơn thuần không reset. Point ở xa được tự mở rộng ρ range và thông báo. Peak đồng hạng được ghi rõ là mơ hồ, không tự chọn một detected line.

## Numerical validation

- `ρ*=√5=2.2360679775`, `θ*=atan2(2,1)=63.434948823°` cho `x+2y=5`.
- Default `A` có shape **20×18**, tổng **162** vote; peak duy nhất **A[14,6]=6**, runner-up **3**.
- Bin thắng: `ρ∈[2,2.5)`, `θ∈[60°,70°)`; detected line `x cos 65° + y sin 65° = 2.25`. Đây là quantization approximation, không trùng line nguồn lý tưởng.
- Incremental vote và batch accumulator cho cùng ma trận; rerun không cộng lại phiếu. Bins ngoài range bị từ chối, không clip vào hàng biên; custom point có range tự mở rộng.
- Preset C dùng line cạnh tranh deterministic; preset D giữ nguyên points và thử bin rộng 30°/1.0. Mọi score của preset được tính lại từ coordinates.
- Full test suite: **39/39 passed** (17 test gốc RANSAC, 4 RANSAC UI regression và 18 Hough tests). Bare-mode Streamlit có cảnh báo ScriptRunContext không phải app exception.

## Browser validation

Edge + Playwright trên Streamlit server thật; health endpoint trả `ok`. Viewports: desktop 1920×1080, laptop 1366×768, tablet 768×1024, mobile 390×844. Kiểm tra Guided initial, point/sinusoid, prediction/reveal, vote từng bước, peak, Lab, đổi bin, preset cạnh tranh, Presentation Mode và quay lại RANSAC. Browser kéo θ tới 178.7° cho ρ âm; đổi Δθ tới 45° và chạy lại accumulator. Một vote với threshold mặc định 4 **không** vẽ detected line trên desktop/tablet/mobile. Không phát hiện Streamlit exception, console/page error hoặc horizontal overflow trong audit. Mobile sidebar Streamlit phải đóng sau khi chọn module để không phủ nội dung.
Browser smoke test bổ sung sau sửa edge case custom point: thêm p10 trùng `(1,2)`, xác nhận p1 gốc không xóa được, chỉ xóa p10; không có browser error.

Ảnh review: [Guided initial](docs/screenshots/hough/desktop-guided-initial.png), [point → sinusoid](docs/screenshots/hough/desktop-point-sinusoid.png), [signed ρ âm](docs/screenshots/hough/desktop-negative-rho.png), [hai curve](docs/screenshots/hough/desktop-two-curves.png), [vote từng bước](docs/screenshots/hough/desktop-partial-vote.png), [final peak](docs/screenshots/hough/desktop-final-peak.png), [Lab sau một vote](docs/screenshots/hough/desktop-lab-vote.png), [lưới coarse](docs/screenshots/hough/desktop-lab-coarse.png), [Presentation Mode](docs/screenshots/hough/desktop-presentation.png), [tablet](docs/screenshots/hough/tablet-lab-vote.png), [mobile](docs/screenshots/hough/mobile-point-sinusoid.png).

## Bugs phát hiện và sửa

- Browser slider cần thao tác trên track; audit ban đầu dùng phím `End` không đổi giá trị. Đã chuyển test sang click track và đọc `aria-valuenow`/`aria-valuetext` để xác nhận thay đổi thật.
- Ở browser, threshold Lab từng khởi tạo bằng 1 do widget mặc định. Đã đặt `value=4` trực tiếp trên widget, bỏ gán session state trước khi tạo widget. Kiểm tra một vote không công bố line.
- Plot kết quả từng hiển thị candidate từ vote cuối cùng cạnh detected line, gây nhiễu. Guided bước 11 chỉ giữ detected bin-center line và ideal source line. Heatmap có ranh giới cell dễ đọc hơn; normal có mũi tên riêng với signed ρ.
- Một custom point trùng tọa độ point gốc từng có thể khiến nút xóa nhầm point gốc. Lab giờ phân biệt bằng chỉ số phần gốc/custom; regression test xác nhận point gốc không thể bị xóa.

## Cleanup

Đã dừng riêng Streamlit server port 8515; health endpoint không còn truy cập được. Tất cả lượt Edge/Playwright của tác vụ đã `browser.close()` và kết thúc; kiểm tra command line không còn tiến trình Edge/Node/Streamlit của tác vụ. Xóa `chapter3_webapp/.hough-audit-tools/` (Playwright cài cục bộ tạm) sau mỗi lượt audit và 8 screenshot thử nghiệm, giữ **14** ảnh review. Tổng file bytes xóa qua hai lượt **28,076,300**. Lượt dọn cuối: dung lượng trống ổ C từ **182,280,138,752** lên **182,294,097,920** byte (tăng **13,959,168** byte); tổng tăng qua hai lượt **28,692,480** byte. Không xóa source, tests, docs, config, Manim hoặc RANSAC.

## Giới hạn và mở rộng

Plotly/Streamlit cập nhật theo từng sự kiện, không là animation 60 fps hoặc direct drag/drop. Lab không tự xử lý ảnh hay phát hiện edge; các point đã biết được dùng để dạy Hough Line. Bin thô/mịn, nhiều line, clutter, threshold đều có thể làm peak khó diễn giải. Peak count là support trong **bin rời rạc**, không chứng minh giao điểm liên tục tuyệt đối hay độ chính xác subpixel. Hướng mở rộng sau review: import edge-point dataset và so sánh nhiều local maxima; không nằm trong phạm vi lần này.
