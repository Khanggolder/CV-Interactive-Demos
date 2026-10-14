# Chapter 2 · Hướng dẫn giảng dạy nhanh

## Mục tiêu và thứ tự

Sau bài học, sinh viên có thể: giải thích vì sao một góc định vị được theo hai hướng; phân biệt scale cục bộ/toàn cục và DoG candidate với SIFT keypoint đã lọc; theo dõi hướng và descriptor 128D; tự tính một quyết định FAST/BRIEF/Hamming; và đánh giá match bằng rule descriptor **lẫn** sai số hình học, precision, coverage.

Thứ tự gợi ý: **Harris/Shi-Tomasi → Gaussian Scale Space/LoG/DoG → SIFT → FAST/BRIEF/ORB-style → Feature Matching Challenge**. Harris trả lời “ở đâu”; scale space trả lời “ở kích thước nào”; SIFT thêm localization, filtering, orientation và description; ORB-style minh họa một chiến lược detector/descriptor khác; Challenge buộc so sánh bằng correspondence đo được. Không suy ra pipeline thắng tuyệt đối từ một fixture.

Kiến thức trước buổi học: ảnh grayscale, convolution/Gaussian, gradient, ma trận 2×2, eigenvalue, tọa độ `[row,col]` so với `(x,y)`, khoảng cách Euclid. Nhắc lại xác suất/seed và affine 2D khi vào Challenge.

## Công thức và thuật ngữ trên bảng

| Chủ đề | Biểu thức/thuật ngữ cần giữ đúng | Lưu ý khi đọc hình |
| --- | --- | --- |
| Harris | `E(u,v)`; `M=Σw∇I∇Iᵀ`; `R=det(M)−k·trace(M)²`; Shi-Tomasi `min(λ₁,λ₂)` | NMS và ngưỡng áp lên response thực; góc gần biên warp không là bằng chứng scale ổn định. |
| Scale Space | `L=Gσ*I`; `σ_increment=√(σ_target²−σ_previous²)`; `σ_global=σ_local·2^octave`; `DoG=L(kσ)−L(σ)` | LoG/DoG có dấu; residual được tăng tương phản *chỉ khi hiển thị*. Cực trị 26 láng giềng vẫn là candidate. |
| SIFT | `Hδ=−g`; histogram hướng 36 bin; descriptor `4×4×8=128`; Lowe `d₁/d₂ < τ` | Một vị trí keypoint có thể sinh nhiều bản ghi hướng/descriptor. Ratio pass không chứng minh đúng hình học. |
| ORB-style | FAST-12 vòng 16; centroid `θ=atan2(m₀₁,m₁₀)`; BRIEF 256 bit; Hamming `popcount(a XOR b)` | Pattern deterministic dùng dạy học, không phải learned OpenCV ORB. Nearest/cross-check/ratio là các rule khác nhau. |
| Challenge | `precision=đúng / accepted evaluable`; `coverage=vị trí A có match đúng / vị trí A có descriptor hợp lệ và quan sát được` | Mẫu số 0 → N/A. Cùng cảnh và ngân sách chưa có nghĩa cùng matching rule. Ground truth chỉ hậu kiểm. |

Giữ các từ chuẩn khi thảo luận: corner detection (phát hiện góc), keypoint (điểm đặc trưng), descriptor (vector mô tả), orientation (hướng), local extrema (cực trị cục bộ), NMS (ức chế cực đại không cục bộ), correspondence (cặp điểm tương ứng), geometric verification (hậu kiểm hình học).

## Điểm dừng trình diễn và câu hỏi dự đoán

1. **Harris:** dùng Guided bước tensor/NMS. Hỏi “cạnh có hai eigenvalue lớn không?” và “đỉnh đồng hạng giữ ai?”. Sinh viên phải chỉ được `M`, `R`, ngưỡng, lân cận 5×5 và KEEP/REJECT trên dữ liệu đang chọn.
2. **Scale Space:** thay `σ`, chọn octave/layer, xem `σ_local × 2^octave`; ở 26-neighbor replay cho sinh viên đoán MAX/MIN/NOT EXTREMUM trước khi reveal. Không gọi số lượng DoG candidate là độ bất biến.
3. **SIFT:** từ candidate DoG qua Taylor/Hessian, orientation và descriptor. Hỏi “hai orientation cùng tọa độ có phải hai vị trí phát hiện khác nhau?”; dùng số đếm hiện tại, không học thuộc con số fixture.
4. **ORB-style:** cho đoán FAST-12 trên vòng intensity thực, sau đó xem centroid, cặp BRIEF và XOR/Hamming. Trước bước matching, yêu cầu phân biệt rule ACCEPT với hậu kiểm đúng/sai hình học.
5. **Challenge:** khóa dự đoán, chạy cùng cặp ảnh, xem các rule và số liệu, thay đúng một biến. Hỏi “nhiều match hơn có luôn tốt hơn?”; chứng minh bằng precision/coverage và một đường match sai cụ thể.

Mỗi lần reveal, yêu cầu sinh viên nói: đầu vào nào đổi, phép tính nào đổi, con số/đường nào là bằng chứng, và giới hạn nào còn lại. Các câu trả lời định lượng phụ thuộc preset/slider hiện tại; không đọc đáp án số cố định từ guide.

## Các nhầm lẫn thường gặp

- Điểm phát hiện nhiều ≠ correspondence tốt. Detector và matching/verification là các khâu khác nhau.
- DoG extremum thô ≠ SIFT keypoint/descriptor đã hoàn thành.
- Ratio/cross-check chấp nhận ≠ đúng hình học; đường trung tính trước reveal chưa mang nhãn đúng/sai.
- Hamming và Euclid khác đơn vị; 32 B so 512 B là payload descriptor, không phải RAM toàn pipeline hay tốc độ thực tế.
- `coverage` ở Challenge không phải detector recall; khi mẫu số rỗng, N/A không phải 100%.
- Invariance là mục tiêu gần đúng; resize/rotation/nhiễu, nội suy và vùng biên có thể làm keypoint biến mất.

## Kịch bản 45 phút

| Phút | Hoạt động |
| --- | --- |
| 0–8 | Harris: dịch cửa sổ, tensor, dự đoán góc/cạnh. |
| 8–16 | NMS và giới hạn một scale → Scale Space Gaussian/DoG. |
| 16–25 | SIFT: candidate → Hessian/hướng/descriptor → ratio. |
| 25–33 | ORB-style: FAST ring → BRIEF/Hamming; đối chiếu chiến lược với SIFT. |
| 33–43 | Challenge: đo cùng cảnh, xem rule, precision/coverage, một failure. |
| 43–45 | Exit question: phân biệt ACCEPT với CORRECT và nêu mẫu số coverage. |

## Kịch bản Lab 90 phút

| Phút | Hoạt động |
| --- | --- |
| 0–10 | Ôn tọa độ, seed, rule, cách lưu minh chứng. |
| 10–25 | Harris Lab: `k`, response và NMS; thử đổi scale, loại điểm gần đệm khỏi lập luận. |
| 25–40 | Scale Space Lab A/B: thay `σ`, octave, LoG/DoG và replay 26 láng giềng. |
| 40–55 | SIFT Lab A/B/C: chọn candidate, orientation/descriptor, ratio, đổi transform. |
| 55–68 | ORB Lab A/B/C: FAST pretest, steering, Hamming/cross-check. |
| 68–85 | Challenge Lab A/B/C: hai thí nghiệm kiểm soát một biến; so rule mặc định và một rule chung; lưu snapshot sau reveal. |
| 85–90 | Trình bày kết luận có số liệu và giới hạn. |

## Bài nộp Feature Matching Challenge

Chọn một cảnh và seed. Ghi dự đoán trước. Chạy ít nhất hai cấu hình chỉ khác **một** biến (góc, scale, ánh sáng, noise/blur, rule hoặc ngưỡng). Báo cáo rule thực sự dùng cho Harris/SIFT/ORB-style; nếu so “cùng policy”, chọn lại cùng rule và chạy lại. Nêu accepted, correct, precision, coverage, descriptor hợp lệ và một match sai hoặc trường hợp N/A. Chú thích một đường match: descriptor chọn vì sao, affine ground truth đặt điểm tương ứng ở đâu, lỗi pixel và ngưỡng. Không suy diễn thứ hạng tốc độ production từ thời gian teaching.

Rubric 10 điểm: kiểm soát biến và tái lập 2; đọc phép tính/mẫu số 3; phân tích một failure với bằng chứng hình học 3; kết luận có điều kiện và giới hạn 2.

## Presentation Mode

Vào từng module rồi bật **Presentation Mode**; bước Guided và trạng thái Lab đang chọn được giữ nguyên. Trên laptop 1366×768 nên cuộn biểu đồ chính lên giữa màn hình, dùng trình duyệt zoom phù hợp máy chiếu; trên mobile dùng layout xếp dọc. Giữ thao tác Prediction → Reveal trực tiếp trước lớp. Tắt Presentation Mode để trở về cùng bước, không phải Reset. Nút Reset của từng module/Lab chỉ dùng khi thật sự muốn bắt đầu lại workspace tương ứng.
