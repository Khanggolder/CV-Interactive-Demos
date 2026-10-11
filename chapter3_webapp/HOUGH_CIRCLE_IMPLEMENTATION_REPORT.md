# Hough Circle Transform · implementation và kiểm chứng

## Phạm vi và file

Module thứ năm trong `chapter3_webapp/`. Không sửa Chapter 1, Manim đã duyệt, engine số học bốn module trước, requirements hoặc framework. Không stage/commit/push/merge.

| File | Vai trò |
| --- | --- |
| `app.py` | Điều hướng module thứ năm, Presentation Mode và CSS mobile/tablet |
| `utils/hough_circle.py` | NumPy-only Cartesian geometry, 6 preset, grid, vote event/state, peak và memory guard |
| `utils/hough_circle_visualization.py` | Hai Plotly panel tỉ lệ trục 1:1; chỉ vẽ từ dữ liệu engine |
| `tutorials/hough_circle.py` | Guided 14 bước, Lab 2 workspace, `circle_*` state, prediction/reveal, vote replay |
| `tests/test_hough_circle.py`, `tests/test_hough_circle_app.py` | 37 numerical/visual + 9 Streamlit integration tests |
| `README.md`, báo cáo này | Tài liệu setup, conventions, giới hạn và QA |
| `docs/screenshots/hough_circle/` | Ảnh Edge/Playwright review |

## Liên tục ≠ rời rạc

Fixture từ `animations/chapter3/hough_circle.py`: tâm `(1.5,0.5)`, r=`2`, các góc `(180,60,300,0,120,240)°`. Sáu điểm thực là `(-0.5,0.5)`, `(2.5,2.2320508076)`, `(2.5,-1.2320508076)`, `(3.5,0.5)`, `(0.5,2.2320508076)`, `(0.5,-1.2320508076)`. Phương trình liên tục `(a-x)²+(b-y)²=r²`; candidate `(a,b)=p+r(cosφ,sinφ)`. Hai điểm đầu cho **hai** nghiệm `(0.5,2.2320508076)` và `(1.5,0.5)`; điểm thứ ba giữ lại duy nhất `(1.5,0.5)`. Đối chiếu sáu point và sáu gradient với `animations/chapter3/media/hough_circle/validation.json` cho sai số lớn nhất **0.0** ở cả hai mảng. Phần này đúng với hình học được Manim verifier kiểm tra.

Manim **không** tính accumulator. Webapp tự thực hiện `A[r_index,b_row,a_col]` int32, center/radius bins nửa mở. Với một r cố định, volume có một lớp và có thể xem như 2D; nhiều r tạo 3D-binned search. Full-locus lấy `angular_samples` điểm trên 360°, nhưng một observation chỉ tăng mỗi cell tối đa một lần **trên một radius layer**; sample trùng bin không tăng support lần nữa. Observation trùng tọa độ cũng không tạo support nhân đôi. Out-of-range bị bỏ, không kẹp vào biên. `Next Vote` tiến đúng một `VoteEvent` và báo rõ có cộng hay không; Previous rebuild từ prefix event để tránh âm/duplicate; Vote Selected/All khớp incremental. Signature gồm point IDs/coordinates/gradients, cả ba tập bin edges, voting mode, samples và polarity, nên đổi cấu hình reset state trước khi cast tiếp. Peak sort ổn định theo count giảm rồi chỉ số cell tăng; tie được hiển thị, không tuyên bố duy nhất. Circle phát hiện dùng đúng `(a,b,r)` ở **tâm bin**, không dùng source center hoặc phép fit ngầm.

Ví dụ chuẩn, center width `.4`, fixed r=`2`, 72 angle samples × 6 point = **432 sample** nhưng chỉ **216 support votes** vì nhiều sample chung bin. Peak duy nhất `A[0,11,13]=6`, bin-center `(1.4,0.6,2)`, sai số tâm `√0.02 ≈ 0.141421`. Known inward gradient chỉ tạo **6 candidate/6 vote**, cùng peak `6`. Đây là số mới của engine rời rạc và được kiểm thử riêng; không gọi chúng là kết quả Manim verifier. Unknown polarity tạo tối đa hai candidate/radius/point, nhưng vẫn dedup nếu cùng bin. Gradient thiếu, zero hoặc không hữu hạn không tạo vote định hướng. Bright-inside intensity `I=.5+.5 tanh((r₀−distance)/.15)` cho inward gradient và `p+r·ĝ`; bright-outside đảo dấu nên dùng `p−r·ĝ`. Cả hai panel dùng Cartesian y-up; raster Sobel/Canny của module cũ vẫn dùng row-down.

Giới hạn an toàn: ≤30 observations, ≤16 radius layers, ≤180 angular samples, ≤240.000 cells (int32 ≤960.000 byte cho volume); ước tính và từ chối trước cấp phát lớn. Event replay lớn nhất theo giới hạn được xây hữu hạn; không tạo hàng triệu Plotly shapes. Dataset/point tùy chỉnh được kiểm tra finite, ID duy nhất và r dương. Đây là detector dạy học trên tọa độ tổng hợp, không phải production circle detector trên ảnh camera.

## Luồng học

Guided: (1) edge/circle, (2) một point, (3) center locus, (4) hai giao điểm + prediction gate, (5) nghiệm chung thứ ba, (6) sáu locus, (7) đổi r, (8) lớp `(a,b,r)`, (9) vote một bin, (10) nhiều point/peak, (11) inward gradient + prediction gate, (12) unknown polarity, (13) circle từ peak bin, (14) giới hạn/recap. Previous/Next/Reset giữ gate; chưa dự đoán không thể reveal/Next ở bước 4 và 11. Presentation tăng figure/marker/label, không đổi engine. Lab A cho đổi point/r/φ/số locus, thêm/bỏ custom point theo ID, chỉ custom mới xóa được. Lab B hỗ trợ known/unknown r, full-locus/known-gradient/unknown-polarity, inward/outward sign, center/radius bin width, angular sampling, Next/Previous Vote, Next Point, Vote Selected, Vote All, Reset, peak inspection và radius slice. Candidate circle sau Next Vote dùng tọa độ từ event thật; heatmap highlight đúng cell. Các control nâng cao nằm trong expander gần hình để mobile đưa hình chính lên cao hơn.

Sáu preset: A chuẩn, B unknown r, C cung thiếu, D outlier, E bright outside/gradient outward, F hai circle. Không nhúng nhãn nguồn vào voting; true circle chỉ dùng vẽ nguồn hoặc báo sai số sau vote. Một peak không mặc định là object thật.

## Tests và browser audit

`python -B -m unittest discover -s tests` trong `chapter3_webapp/`: **139/139 passed** (93 baseline + 46 mới), không failures. Numerical tests kiểm tra point/gradient theo công thức độc lập, finite difference, giao hai locus/tangent/disjoint/trùng, ba điểm suy biến, tọa độ âm, bin nửa mở, out-of-range, dedup angular/observation, independent NumPy reference cho count, incremental=batch, state invalidation, polarity, unknown-radius, competing circles, memory guard và circle shape. AppTest kiểm tra năm module, Guided end-to-end/gates/reset, vote rerun không nhân đôi, preset/custom IDs, Lab parameter invalidation, unknown radius, zero-gradient custom point và Presentation. `missing ScriptRunContext` là warning bare-mode của Streamlit testing, không phải app exception.

Server Streamlit thật trên localhost port 8525: health endpoint `ok`. Edge headless + Playwright trên desktop **1920×1080**, laptop **1366×768**, tablet **768×1024**, mobile **390×844**. Browser hoàn thành Guided 14 bước, hai gates, Previous/Reset, vote partial/full, Lab/Presentation và chụp các viewport. Lab browser kiểm thêm unknown-radius (**3024** sample → **1410** support), gradient (**42/42**), unknown polarity (**84** candidate → **81** support do dedup), đổi center/radius bin reset về **0**, preset F hai circle (**3456** sample → **1429** support, peak 6 trong cấu hình thử), preset E bright-outside có Outward được chọn. Với center bin `.5`, float vote gần ranh bin chia peak `5`; trở về `.4` cho peak `6/6`, minh họa quantization thật chứ không thay bằng truth. Kiểm tra DOM không có `stException`, page/console error hay horizontal overflow. Ảnh đã mở xem trực tiếp; kiểm tra đúng phép toán dựa trên numerical tests, không suy từ screenshot. UI giới hạn cấu hình tối đa an toàn nên oversized accumulator được kiểm bằng numerical test, không có form browser để yêu cầu cấp phát vô hạn; coincident-loci degeneracy cũng được kiểm bằng engine test.

Ảnh đại diện: [initial](docs/screenshots/hough_circle/desktop-guided-initial.png), [candidate](docs/screenshots/hough_circle/desktop-moving-candidate.png), [two intersections](docs/screenshots/hough_circle/desktop-two-intersections.png), [three points](docs/screenshots/hough_circle/desktop-three-point-center.png), [radius layers](docs/screenshots/hough_circle/desktop-radius-layers.png), [first vote](docs/screenshots/hough_circle/desktop-first-vote.png), [peak](docs/screenshots/hough_circle/desktop-center-peak.png), [gradient](docs/screenshots/hough_circle/desktop-gradient-inward.png), [detected](docs/screenshots/hough_circle/desktop-detected-bin.png), [unknown r](docs/screenshots/hough_circle/desktop-unknown-radius-accumulator.png), [competing](docs/screenshots/hough_circle/desktop-competing-circles.png), [outward](docs/screenshots/hough_circle/desktop-outward-polarity-preset.png), [Lab](docs/screenshots/hough_circle/desktop-lab-full-vote.png), [Presentation](docs/screenshots/hough_circle/desktop-presentation.png), [tablet](docs/screenshots/hough_circle/tablet-guided-initial.png), [mobile Guided](docs/screenshots/hough_circle/mobile-guided-initial.png), [mobile Center Space](docs/screenshots/hough_circle/mobile-guided-center-space.png), [mobile Lab](docs/screenshots/hough_circle/mobile-lab-voting-chart.png).

## Defects sửa trong audit và giới hạn còn lại

- Bước 1 ban đầu lộ candidate circle trước lúc giảng locus; đã ẩn candidate ở bước mở đầu.
- Bước 1 mobile còn Center Space trống chiếm một màn hình; đã chuyển bước này sang Image Space đơn, hai panel bắt đầu ở bước 2.
- Radius slice từng hiển thị heatmap lớp r khác candidate circle; đã đồng bộ radius của hai hình với lớp đang xem.
- `Next Vote` trong Lab từng đổi accumulator nhưng candidate circle còn theo φ slider cũ; đã lấy vị trí từ `VoteEvent` thật.
- Browser thật ban đầu đẩy slider φ/r và sampling/bin về mức nhỏ nhất khi widget xuất hiện muộn hoặc trong expander; đã tách model khỏi slider φ/r và khởi tạo rõ default của select-slider. Kiểm lại vote đầu từ p1 → `(1.5,0.5)` ở r=`2` và 72 sample, không phải r=`0.5`/12 sample.
- Custom point/gradient number inputs mở ở `−10`; đặt default `0`, test zero-gradient không cast vote.
- Với radius slice khác detected cell, thêm chú thích hai lớp r khác nhau. Tie count từng bị giới hạn 10; giờ báo mọi ô đồng hạng, chỉ danh sách inspect cắt còn top 10.
- Lab mobile đặt nhiều setup controls trước hình; đã thu vào expander, đưa chart trước phần diễn giải peak dài.
- Streamlit sidebar tự đóng trên tablet/mobile; audit browser mở/đóng đúng trước khi kiểm tra, không ép sidebar luôn mở.

Hạn chế: góc full-locus chỉ được sampling hữu hạn nên có thể bỏ lỡ bin hữu ích; bin center gây lượng tử center/radius; outlier/thiếu cung/hai circle có thể làm peak mơ hồ. Unknown polarity tăng candidate count và sai polarity mất tâm đúng; gradient nhập tay có thể sai. Không có image edge extraction, subpixel refinement, multi-circle non-maximum suppression, drag/drop hay detector ảnh sản xuất. Ở lab, true outline của synthetic preset là gợi ý thị giác; số peak không dùng nhãn này.

## Cleanup và Git

Đã đóng browser contexts/Edge headless và dừng riêng server Streamlit port 8525; không còn listener ở cổng này. Đã xóa đúng thư mục công cụ tạm `.circle-audit-tools/` (121 file, 13.669.591 byte), 8 ảnh thử nghiệm trùng/dư (1.040.785 byte), 4 file Python bytecode tạo trong audit (160.748 byte), và một profile Playwright tạm rỗng của riêng tác vụ. Tổng dung lượng file xóa: **14.871.124 byte (~14,87 MB)**; giữ **26 ảnh review** (3.318.999 byte) trong `docs/screenshots/hough_circle/` cùng toàn bộ source, tests và tài liệu. Dung lượng trống ổ C lúc kiểm tra cuối: **186.109.100.032 byte**; không quy chênh lệch dung lượng trống toàn ổ cho tác vụ vì tiến trình khác có thể ghi đĩa đồng thời.

Git cuối: đang ở `feature/chapter3-manim-animations`; `git status --short --branch` chỉ có `?? chapter3_webapp/`. Không stage, commit, push hoặc merge. Chapter 1 và các Manim animation đã duyệt không thay đổi.
