# Chapter 3 · RANSAC + Hough Line + Sobel + Canny + Hough Circle + LSD

Ứng dụng Streamlit gồm sáu module độc lập: RANSAC line fitting, Hough Line Transform + Accumulator, Sobel Convolution, Canny Edge Detector, Hough Circle Transform và LSD Line Segment Detector. Người học quan sát, dự đoán, thao tác, tính toán thật và giải thích giới hạn.

Các scene Manim đã duyệt vẫn được giữ nguyên; app không cần Manim runtime.

## Teaching philosophy

Luồng học chính là:

`Predict → Interact → Observe → Explain → Experiment`

- **Guided Tutorial** đi qua 10 bước: point cloud, dự đoán, sample, candidate, khoảng cách vuông góc, dải epsilon, consensus, lặp, TLS refit và so sánh OLS.
- **Experiment Lab** cho phép tạo dữ liệu và chạy một hoặc nhiều iteration. Mỗi trial có năm pha xem lại từ trạng thái thật: Sample → Fit → Measure → Consensus → Update Best.
- Ba preset tập trung vào clean data, many outliers và wrong epsilon. Preset cuối có cả biến thể epsilon quá nhỏ và quá lớn.
- **Presentation Mode** tăng kích thước biểu đồ, marker và nhãn khi trình chiếu; chế độ thường và mobile vẫn giữ bố cục riêng.
- **Hough Guided Tutorial** gồm 12 bước: edge points → normal form → sinusoid → giao điểm → discretization → voting → peak/bin-center line → giới hạn. Bước giao điểm có prediction/reveal gate; các vote là số nguyên được cộng từng hành động.
- **Hough Experiment Lab** có hai workspace dùng chung point dataset: Point-to-Sinusoid Explorer và Accumulator Voting Explorer. Có custom point, đổi bin width, vote một bước/một point/toàn bộ, reset và bốn teaching preset. State Hough và RANSAC được namespaced riêng.
- **Sobel Guided Tutorial** gồm 12 bước: pixel → patch → Gx mask → 9 tích → Gx → Gy trên cùng patch → vector → M/θ → gradient map → các hướng biên → noise → giới hạn. Gy và kết luận có prediction/reveal gate.
- **Sobel Experiment Lab**: Pixel & Kernel Explorer cho inspect/edit intensity 0–255, di chuyển center từng ô, chọn Gx/Gy, tính từng tích và reset; Gradient Explorer xem bản đồ Gx, Gy, magnitude, direction thật và vector tại ô đang chọn. Presentation Mode tăng kích thước ma trận/nhãn. State dùng tiền tố `sobel_*`.
- **Canny Guided Tutorial** gồm 14 bước theo dõi *một ảnh duy nhất*: original → Gaussian smoothing → signed Sobel gradient → magnitude/direction → NMS KEEP/SUPPRESS → double threshold → 8-connected hysteresis → final binary map. Các gate yêu cầu dự đoán trước khi reveal NMS và threshold. Pixel inspector 7×7 giữ cùng tọa độ qua từng stage; BFS replay dùng event thật.
- **Canny Experiment Lab** có Pipeline Explorer (9 stage, 6 preset deterministic, σ/low/high, pixel đồng bộ) và NMS & Hysteresis Explorer (q1/p/q2, 4 sector, BFS checkpoint, connected/isolated weak). Presentation Mode tăng hình và chữ, ẩn bớt thông tin phụ; state dùng `canny_*`.
- **Hough Circle Guided Tutorial** gồm 14 bước: một edge point → locus tâm → hai nghiệm có thể → điểm thứ ba → radius layers → voting 2D/3D-binned → gradient và polarity → detected circle từ bin center → giới hạn. Gate yêu cầu dự đoán trước khi reveal hai giao điểm và gradient vote.
- **Hough Circle Experiment Lab** có Center Locus Explorer và Circle Voting Explorer trên cùng dataset; sáu preset deterministic, custom point/gradient, vote từng candidate/một point/toàn bộ, rewind/reset, peak/ties, radius slice và Presentation Mode. State dùng `circle_*`.
- **LSD Guided Tutorial** gồm 14 bước: intensity → gradient → level line → strongest seed → axial 8-connected BFS → weighted-PCA finite segment/rectangle → directed binomial NFA → accepted segments. Seed và NFA có predict/reveal gate; BFS xem từng comparison thực và tua lại.
- **LSD Experiment Lab** có Gradient & Region Explorer và Segment & NFA Explorer trên cùng ảnh và tham số; sáu preset deterministic, manual seed riêng, τ/minimum magnitude/ε, candidate selection, replay, Presentation Mode. State dùng `lsd_*`.

## Architecture

```text
chapter3_webapp/
├── .streamlit/config.toml
├── app.py
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── ransac.py
│   ├── visualization.py
│   ├── hough.py
│   ├── hough_visualization.py
│   ├── sobel.py
│   ├── sobel_visualization.py
│   ├── canny.py
│   ├── canny_visualization.py
│   ├── hough_circle.py
│   ├── hough_circle_visualization.py
│   ├── lsd.py
│   └── lsd_visualization.py
├── tutorials/
│   ├── hough.py
│   ├── sobel.py
│   ├── canny.py
│   ├── hough_circle.py
│   └── lsd.py
└── tests/
    ├── test_app.py
    ├── test_ransac.py
    ├── test_visualization.py
    ├── test_hough.py
    ├── test_hough_app.py
    ├── test_sobel.py
    ├── test_sobel_app.py
    ├── test_canny.py
    ├── test_canny_app.py
    ├── test_hough_circle.py
    ├── test_hough_circle_app.py
    ├── test_lsd.py
    └── test_lsd_app.py
```

Các numerical engine trong `utils/` không phụ thuộc Streamlit/Plotly/Manim. UI và biểu đồ chỉ hiển thị số học đã tính.

## Mathematical assumptions

Candidate line dùng dạng implicit đã chuẩn hóa:

```text
a*x + b*y + c = 0
sqrt(a*a + b*b) = 1
```

Khoảng cách vuông góc:

```text
d = abs(a*x + b*y + c) / sqrt(a*a + b*b)
```

Điểm thuộc consensus khi `d <= epsilon`. Phép so sánh cho phép dung sai số học `1e-12` để các điểm sample vẫn nằm trên chính đường thẳng của chúng khi `epsilon = 0`. Trial có consensus lớn nhất thắng; tổng bình phương residual trong consensus chỉ dùng để phá hòa. Final model được refit bằng Total Least Squares / PCA trên toàn bộ winning consensus set. Nhãn dùng để sinh dữ liệu không tham gia sampling, scoring hoặc fitting.

Ordinary Least Squares trong màn hình so sánh là hồi quy `y = m*x + b`, tối thiểu hóa vertical residual. Với dữ liệu có mọi hoành độ giống nhau, app dùng đường đứng `x = mean(x)` để tránh model không hợp lệ.

## Setup and run

Từ repository root trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run chapter3_webapp\app.py
```

Root `requirements.txt` là manifest canonical cho **một** `.venv` dùng chung Chapter 1, Chapter 3 và Manim. Không tạo môi trường con. Trong sidebar, chọn một trong sáu module, rồi Guided hoặc Lab.

## Hough conventions và fixture

Image Space dùng hệ Cartesian với y tăng lên trên. θ là góc của **normal** trong `[0°,180°)`, ρ có dấu:

```text
ρ = x cos θ + y sin θ
x cos θ + y sin θ = ρ
```

Fixture mặc định gồm 6 point trên `x + 2y = 5`: `(1,2)`, `(3,1)`, `(0.2,2.4)`, `(1.8,1.6)`, `(3.7,0.65)`, `(4.5,0.25)`; và 3 distractor: `(0.5,0.3)`, `(2.5,3.3)`, `(4.1,2.6)`. Giao điểm Hough liên tục là `(θ*,ρ*)=(atan2(2,1),√5) ≈ (63.435°,2.236)`. Mỗi point tạo một sinusoid liên tục; giao điểm chung biểu thị một line đi qua nhiều point.

Accumulator mặc định có 18 θ bin rộng 10° và 20 ρ bin rộng 0.5 trên `[-5,5)`. Mỗi point vote một lần ở tâm mỗi θ bin; `A[rho_row,theta_column]`. Chín point tạo **162 vote**, peak duy nhất `A[14,6]=6`, runner-up 3. Peak nằm trong `θ∈[60°,70°)`, `ρ∈[2,2.5)`; detected line dùng **bin center** `θ=65°`, `ρ=2.25`, không phải line nguồn chính xác. Thêm point xa sẽ mở rộng ρ range đối xứng; đổi points hoặc bin width sẽ reset vote state để không trộn cấu hình.

## Tests

```powershell
Push-Location chapter3_webapp
..\.venv\Scripts\python.exe -B -m unittest discover -s tests
Pop-Location
```

Tests kiểm tra cả sáu module, gồm Hough Circle geometry/voting và LSD gradients, axial BFS, weighted-PCA, directed NFA, state invalidation, Guided/Lab/Presentation và module switching. Kết quả hiện tại: **175/175 passed** (139 baseline + 36 LSD).

## Visual verification and screenshots

1. Chạy app ở kích thước desktop và kiểm tra Guided Tutorial từ bước 1 đến 10.
2. Ở bước 4, xác nhận các đoạn residual vuông góc với candidate và dải epsilon có cùng tỷ lệ x/y.
3. Ở bước 8 và 9, xác nhận TLS refit và OLS có legend riêng.
4. Trong Experiment Lab, chạy từng preset; sau `Chạy 1 iteration`, đi qua năm pha rồi thử `Chạy thêm 10` và `Chạy toàn bộ N`.
5. Kiểm tra ở viewport hẹp: step indicator gọn một hàng và các nút Previous, Next, Reset vẫn dùng được.
6. Bật Presentation Mode và xác nhận biểu đồ, marker, nhãn lớn hơn nhưng công thức/kết quả không đổi.

Kết quả browser audit MVP ở [UI_AUDIT.md](UI_AUDIT.md); đợt tinh chỉnh hiện tại và screenshot ở [UI_REFINEMENT.md](UI_REFINEMENT.md) và `docs/screenshots/refinement/`.
Hough implementation và browser audit: [HOUGH_IMPLEMENTATION_REPORT.md](HOUGH_IMPLEMENTATION_REPORT.md), screenshots ở `docs/screenshots/hough/`.

## Sobel conventions và cách chạy

Trong sidebar chọn **Sobel Convolution**, rồi Guided hoặc Experiment Lab; bật **Presentation Mode** khi trình chiếu. Không thêm dependency: chạy `python -m streamlit run chapter3_webapp/app.py` từ repo root, và `python -m unittest discover -s tests` trong `chapter3_webapp/` bằng environment đã cài `requirements.txt`.

Ảnh grayscale dùng số nguyên 0–255. Mask Gx `[-1,0,1];[-2,0,2];[-1,0,1]`, Gy `[-1,-2,-1];[0,0,0];[1,2,1]` được **áp trực tiếp theo cross-correlation**, không lật như convolution toán học. Hàng/y ảnh tăng xuống, cột/x tăng sang phải. `Gx=sum(P*Kx)`, `Gy=sum(P*Ky)`, `M=hypot(Gx,Gy)`, `θ=atan2(Gy,Gx)` theo tọa độ ảnh. Vector khi vẽ lên mặt phẳng y hướng lên là `(Gx,-Gy)`. Nếu M=0, θ **Undefined**. Response Gx/Gy có dấu, không clip 0–255; màu Plotly chỉ chuẩn hóa để nhìn và nhãn ô vẫn là giá trị thô. Chỉ tính center interior của patch 3×3, border hiển thị `—`, không zero-padding.

Fixture mặc định 5×7 có 20 bên trái, 200 từ cột 4 trở đi; center `(2,3)` cho patch ba hàng `[20,20,200]`, Gx=720, Gy=0, M=720, θ=0°. Sáu preset deterministic: vertical, horizontal, diagonal, flat, reversed contrast, noisy edge fixed-seed. Gradient đo thay đổi intensity cục bộ, không chứng minh semantic boundary; noise/texture/corner có thể gây response và Sobel không có smoothing, non-maximum suppression hay hysteresis. Xem [SOBEL_IMPLEMENTATION_REPORT.md](SOBEL_IMPLEMENTATION_REPORT.md) và `docs/screenshots/sobel/` cho audit.

## Canny conventions và cách học

Chọn **Canny Edge Detector** trong sidebar, sau đó **Guided Tutorial** hoặc **Experiment Lab**; bật **Presentation Mode** khi giảng bài. Ảnh mặc định là generator 224×224 của scene `animations/chapter3/canny_pipeline.py`: circle, rectangle, diagonal, ring, texture và noise với seed 13. Năm preset bổ sung: clean geometry, noisy geometry (seed 29), connected weak, isolated weak, fine detail/blur. Mỗi preset có câu hỏi dự đoán, tham số điều chỉnh và kết quả tính từ input thật.

- Original uint8 được đổi sang intensity float `[0,1]`. Gaussian 5×5, σ mặc định **1.15**, trọng số tính từ công thức và chuẩn hóa tổng 1; border dùng `np.pad(..., mode="reflect")` (không phải edge/zero padding).
- Sobel Kx/Ky như scene được áp **cross-correlation**, không lật. Hàng ảnh/y tăng xuống; Gx/Gy có dấu. `M_raw=hypot(Gx,Gy)`, `θ=atan2(Gy,Gx)`; M=0 thì hướng **Undefined** về mặt ngữ nghĩa. Plotly y hướng lên sẽ cần đảo dấu Gy khi vẽ vector. `M_norm=clip(M_raw/max(P99.5(M_raw),1e−9),0,1)`; raw và normalized không cùng thang.
- NMS so sánh `M_norm` với q1/q2 theo 4 sector **0°, 45°, 90°, 135°** trong tọa độ ảnh; chỉ giữ khi `M(p)≥M(q1)` **và** `M(p)≥M(q2)`, border một pixel bằng 0. Đây là NMS lượng tử 4 hướng, **không** nội suy subpixel; plateau đồng hạng có thể còn nhiều pixel. Sau NMS, chuẩn hóa lại `NMS_norm=NMS/max(max(NMS),1e−9)`.
- Double threshold áp trên **NMS_norm**, mặc định `low=0.10`, `high=0.35`, yêu cầu `0<low<high≤1`. Strong `≥high`; weak `low≤value<high`; zero background không được nhận. Hysteresis BFS **8-connected** bắt đầu từ mọi strong và chỉ lan qua weak; accepted weak vẫn mang lớp weak. Final map nhị phân; replay event cuối trùng batch map.
- Replay giữ **thứ tự BFS đầy đủ**, nhưng Next/Previous nhảy qua những queue pop không phát hiện weak để hình thay đổi có ý nghĩa. Caption ghi số event thật; Run to completion vẫn xử lý toàn bộ. Đổi ảnh, σ hoặc threshold sẽ reset replay/reveal, đổi stage hay pixel không tính lại pipeline. Cache gồm image bytes, kích thước, σ, low/high tương ứng.

Giới hạn: NMS 4 hướng có sai số lượng tử và tie plateau; 8-connectivity có thể nối nét không mong muốn; smoothing mạnh làm mất chi tiết; threshold cố định không phù hợp mọi ảnh. Đây là implementation giảng dạy bám scene, **không** thay toàn bộ bằng `cv2.Canny()`. Xem [CANNY_IMPLEMENTATION_REPORT.md](CANNY_IMPLEMENTATION_REPORT.md) và `docs/screenshots/canny/`.

## Hough Circle conventions và cách học

Chọn **Hough Circle Transform** trong sidebar. Ví dụ Manim chuẩn giữ tâm `(1.5,0.5)`, bán kính `2`, sáu point ở các góc `180°,60°,300°,0°,120°,240°`, tọa độ Cartesian `y↑`. Một point `p=(x,y)` với r cố định tạo locus tâm `(a,b)=p+r(cosφ,sinφ)`; hai point đầu có **hai** giao điểm hợp lệ, điểm thứ ba giữ lại `(1.5,0.5)`. `I=0.5+0.5 tanh((r₀−distance)/0.15)` sáng trong/tối ngoài, gradient ở biên hướng vào nên vote đúng dấu là `p+r·ĝ`. Đảo tương phản thì gradient hướng ra và phải dùng `p−r·ĝ`; polarity chưa biết thì xét cả hai. Gradient zero/thiếu không được suy ra từ nhãn nguồn.

Manim chỉ minh họa locus **liên tục**. Webapp thêm accumulator hữu hạn `A[r_index,b_row,a_col]`: grid center và radius gồm các bin nửa mở; full-locus sample các góc, mỗi observation chỉ ủng hộ một bin tối đa một lần trên mỗi radius layer. `Next Vote` có thể ghé cùng ô mà **không tăng count**. Detected circle lấy từ **bin center** của một peak; tie được báo rõ, không thay bằng ground truth hoặc fit liên tục. Unknown-radius dùng nhiều lát cắt chung center grid; chuyển r/bin/sampling/polarity/point dataset reset vote state. Giới hạn hiện tại: tối đa 30 point, 16 radius layers, 180 angle samples, 240.000 accumulator cells (int32); cấu hình vượt hạn mức bị từ chối trước khi cấp phát.

Sáu preset: chuẩn, chưa biết r, cung thiếu, outlier, đảo polarity, hai circle cạnh tranh. Lab A điều khiển φ/r/point/loci và thêm/bỏ custom point theo ID; Lab B vote từng sample, một point hoặc tất cả, rewind/reset, xem peak và radius layers. Các chart đều dùng Cartesian y-up và tỉ lệ trục 1:1; ảnh raster của Sobel/Canny vẫn dùng row hướng xuống. Xem [HOUGH_CIRCLE_IMPLEMENTATION_REPORT.md](HOUGH_CIRCLE_IMPLEMENTATION_REPORT.md) và `docs/screenshots/hough_circle/`.

## LSD teaching prototype · conventions và cách học

Chọn **LSD Line Segment Detector** trong sidebar. Guided Tutorial gồm 14 bước có predict/reveal và BFS replay; Experiment Lab có hai workspace dùng chung dataset/τ/minimum magnitude/ε. Presentation Mode tăng kích thước grid/nhãn/NFA. Sáu preset deterministic gồm reference, clean edge, noisy texture, short fragments, crossing structures và contrast reversal. UI không gọi OpenCV `createLineSegmentDetector()` và không cần Manim runtime.

Fixture approved giữ nguyên `DATA_SEED=9`, shape `(10,12)`, chỉ số `[y,x]` zero-based và **y tăng lên**. `Gy,Gx=np.gradient(I)`, `M=hypot(Gx,Gy)`, `θlevel=(atan2(Gy,Gx)−π/2) mod 2π`. Seed là pixel chưa dùng có M lớn nhất, tie theo row-major. Region growing 8-connected dùng `<τ` trên góc **axial modulo π** với hướng seed cố định. Weighted PCA fit tangent; endpoints lấy projection support min/max (đoạn hữu hạn), rectangle đệm nửa pixel-square. NFA dùng alignment **directed modulo 2π**, `p₀=τ/π`, `N_T` bằng số LSR đã grow trên toàn ảnh, và `NFA=N_T·P(Binomial(n,p₀)≥k)`. ACCEPT khi `NFA<ε`.

Reference τ=22.5°, minimum M=0.09, ε=1 tái tạo **N_T=20**, Candidate A `n=33,k=33,NFA≈3.1554436e−29` (ACCEPT) và B `n=4,k=2,NFA≈1.5771484` (REJECT). Đổi τ/M tạo lại toàn bộ vùng và score; đổi ε cập nhật quyết định. Manual seed chỉ để khám phá một vùng, không ghép với NFA full-image. NFA dùng log-domain upper tail để tránh overflow/underflow trung gian, và hiển thị scientific notation khi rất nhỏ.

**Đây là interactive LSD teaching prototype dựa trên Chapter 3 LSR/binomial NFA, không phải published reference LSD đầy đủ. NFA hiển thị không phải bảo đảm false-alarm toàn ảnh đã hiệu chuẩn.** Vùng được chọn thích nghi và `N_T` chỉ đếm LSR đã grow, không phải toàn bộ rectangle-test family. Xem [LSD_IMPLEMENTATION_REPORT.md](LSD_IMPLEMENTATION_REPORT.md) và [ảnh browser audit](docs/screenshots/lsd/).

## Known limitations

- MVP chỉ fit một đường thẳng 2D mỗi lần chạy.
- App không triển khai drag-and-drop hoặc direct Plotly point selection vì callback selection dễ làm flow giảng dạy thiếu ổn định; sample được tạo từ PRNG có seed và hiển thị rõ chỉ số điểm.
- Outlier ngẫu nhiên có thể tình cờ nằm gần cấu trúc chính. Đây là hành vi hợp lệ của phép phân loại theo khoảng cách.
- Số iteration hữu hạn không bảo đảm tìm được sample toàn inlier.
- Epsilon phải do người dùng chọn dựa trên thang đo và mức nhiễu.
- OLS và TLS tối ưu hai loại residual khác nhau; phần so sánh ghi rõ distinction này.
- Chưa có nhập point cloud từ file và chưa triển khai multi-model RANSAC.
- Hough Line chỉ tìm line normal-form từ point dataset 2D; chưa có line segment extraction/direct drag-drop. Hough Circle là prototype trên edge points tổng hợp/nhập tọa độ, chưa tự chạy detector trên ảnh thật hoặc tinh chỉnh subpixel; bin thô, sampling thưa, thiếu cung, clutter và nhiều circle đều có thể tạo peak sai/mơ hồ.
