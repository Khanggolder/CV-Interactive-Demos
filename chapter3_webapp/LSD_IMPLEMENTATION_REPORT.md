# LSD Line Segment Detector · implementation và kiểm chứng

## Phạm vi, cấu trúc

Module thứ sáu trong `chapter3_webapp/`, nối vào selector/Guided/Lab/Presentation hiện có; không tạo app mới. Chỉ sửa trong `chapter3_webapp/`. Không sửa Chapter 1, Manim/video đã duyệt, năm engine cũ, `requirements.txt` hoặc Git branch. Không stage/commit/push/merge.

| File | Vai trò |
| --- | --- |
| `app.py` | Điều hướng module thứ sáu, CSS stepper/mobile và Presentation |
| `utils/lsd.py` | NumPy-only fixture, gradients, axial BFS, replay, weighted PCA, binomial log-tail/NFA, six presets |
| `utils/lsd_visualization.py` | Plotly intensity/gradient/level-line, region, segment, rectangle, NFA; Cartesian y↑ |
| `tutorials/lsd.py` | Guided 14 bước, 2 Lab workspace, `lsd_*` state, prediction/reveal và invalidation |
| `tests/test_lsd.py`, `tests/test_lsd_app.py` | 27 numerical/visual + 9 Streamlit interaction tests |
| `README.md`, báo cáo này | Conventions, setup, giới hạn và audit |
| `docs/screenshots/lsd/` | Ảnh Edge/Playwright đã mở xem trực quan |

## Quy ước toán học và fixture

Input `[y,x]` zero-based với **y tăng lên**. Fixture Manim seed 9 là 10×12:

`I=.5+.42*tanh((y-.55*x-1.3)/.8)+.40*exp(-((x-8.8)^2+(y-1.6)^2)/(2*.75^2))+Normal(0,.055)`.

Không clip input; heatmap chỉ chuẩn hóa display. `Gy,Gx=np.gradient(I)`, `M=hypot(Gx,Gy)`, `θg=atan2(Gy,Gx)`, `θlevel=(θg−π/2) mod 2π`. Gradient zero không có hướng hình học và không được chọn làm seed. Seed là gradient eligible mạnh nhất chưa dùng, tie row-major. BFS 8-neighbor giữ hướng seed ban đầu; chỉ nhận neighbor nếu `M≥minimum` và `angular_difference_pi(θneighbor,θseed)<τ` (strict). Event lưu parent, candidate, M, góc, Δaxial, τ, quyết định và lý do; replay từ prefix event tái tạo đúng member set, không cộng lại trên rerun. Pixel visited/used/dưới ngưỡng bị bỏ trước phép so góc như prototype.

Weighted centroid `Σ(M_i·p_i)/ΣM_i`; dominant eigenvector của `ΣM_i(p_i−center)(p_i−center)^T` là tangent, hướng dấu theo tổng level-line directed có trọng số. Một pixel/degenerate scatter dùng hướng seed an toàn. Endpoints là min/max projection của pixel centers trên tangent, **đoạn hữu hạn**. Rectangle dùng padding `0.5(|tx|+|ty|)` và `0.5(|nx|+|ny|)` để bao các pixel squares, không chỉ tâm. Plotly khóa tỉ lệ x/y 1:1.

NFA dùng **directed** `angular_difference_2pi(θmember,θsegment)<τ`, khác phép axial modulo π dùng khi grow. `n` là số pixel LSR; `k` là số pixel aligned directed; `p₀=τ/π`; `N_T` là số candidate LSR từ scan toàn ảnh; `NFA=N_T·Σ[j=k..n] C(n,j)p₀^j(1−p₀)^(n−j)`; ACCEPT khi `NFA<ε`. Tail tính qua `lgamma`/log-sum-exp để tránh overflow trung gian; xác suất dưới float range giữ log để hiển thị dạng scientific thay vì làm như thể bằng 0. Giá trị `n=0`, `k=0`, `k=n`, p gần 0/1 và input không hợp lệ có test.

**This is an interactive LSD teaching prototype based on the Chapter 3 LSR and binomial NFA model. It is not the full published reference LSD implementation, and the displayed NFA is not a calibrated whole-image false-alarm guarantee.** LSR được chọn thích nghi; `N_T` chỉ là số vùng đã grow, không phải toàn bộ rectangle-test family của paper. NFA nhỏ chỉ là bằng chứng mạnh hơn theo mô hình đã nêu; ACCEPT không chứng minh có biên vật thể vật lý.

## Đối chiếu approved Manim

Đối chiếu trực tiếp engine webapp với `animations/chapter3/lsd.py` (read-only import) và `media/lsd/validation.json` (read-only): năm mảng `intensity`, `gx`, `gy`, `magnitude`, `angles` có max absolute error **0**; toàn bộ thứ tự seed/membership, `k`, endpoints và rectangle trùng; NFA sai khác lớn nhất **3.55×10⁻¹⁵** do đường tính tail khác. Reference `τ=22.5°`, min M `0.09`, `ε=1`:

| Candidate | Seed (y,x) | n | k | NFA | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| A / #1 | (6,8) | 33 | 33 | 3.155443620884e−29 | ACCEPT |
| B / #2 | (1,0) | 4 | 2 | 1.5771484375 | REJECT |

**N_T=20**; hai candidate trên là ví dụ trong 20 vùng, không phải cả phép quét. Tất cả score/decision/UI lấy từ engine, không hardcode các n/k/nhãn này. Giá trị khi ε đổi từ 1 lên 5: B vẫn NFA≈1.57715 nhưng chuyển thành ACCEPT đúng bất đẳng thức.

Sáu preset deterministic cho (candidate count, accepted count) ở default: reference `(20,3)`, clean strong line `(5,5)`, noisy texture `(57,5)`, short fragments `(16,0)`, crossing structures `(5,4)`, global contrast reversal `(20,3)`. Không gán nhãn kết quả cố định cho preset. Global reversal đảo cả gradient và tangent fit nên n/k có thể giữ nguyên; nó không phải ví dụ pha trộn polarity trong cùng LSR.

## Luồng học và state

Guided có 14 bước từ intensity đến kết luận. Pixel inspector hiện Gx/Gy/M/góc thật. Bước seed yêu cầu dự đoán trước reveal. BFS có Start/Next/Previous/Run Several/Complete/Reset trên event thật; bước 6 cần ít nhất một comparison và bước 7 phải complete trước khi Next. Bước 8/9 fit đoạn và rectangle, bước 11/12 buộc dự đoán → reveal phép tính → reveal decision. Màu candidate trung tính trước decision, tránh lộ đáp án qua title/đường vẽ. Bước 13 hiển thị 20 candidate và các đoạn accepted; bước 14 nêu giới hạn và concept check.

Lab A cho đổi intensity preset, chọn pixel/view, chọn strongest hoặc **manual exploration**, step/replay LSR và xem fit. Manual seed không được ghép NFA toàn ảnh của scan khác. Lab B chọn computed candidate, xem alignment cyan/cam, thay τ/M/ε, tính NFA và so mọi candidate sau reveal. Đổi τ/M/dataset tái tính region/fit/k/p₀/N_T/NFA và reset replay/reveal; đổi ε không thay hình học/NFA nhưng reset decision reveal. Chọn candidate khác xóa reveal cũ. `lsd_*` không đụng state năm module trước.

Presentation Mode tăng grid, marker, metric; NFA đặt cùng hàng với hình ở desktop. Với laptop/mobile biểu đồ và công thức vẫn không tràn ngang. Lab settings được thu gọn mặc định, nên chart xuất hiện sớm hơn trên mobile; nội dung toán vẫn đầy đủ trong expander. Không có điều khiển giả hay animation tự chạy.

## Test và browser audit

Chạy `..\.venv\Scripts\python.exe -B -m unittest discover -s tests` từ `chapter3_webapp/`: **175/175 passed** (139 baseline + 36 LSD), không fail/error. Bare-mode AppTest có cảnh báo `missing ScriptRunContext` quen thuộc, không phải app exception. Test mới gồm fixture độc lập, gradient/góc/zero, axial-vs-directed và strict boundary, row-major seed, BFS/fixed reference/used mask, replay/reset, PCA/rectangle, stable binomial, A/B/N_T, parameter invalidation, presets, no state collision, Guided 14-step gates, Lab và Presentation. Không xóa/sửa test cũ. `git diff --check` không đủ vì `chapter3_webapp/` untracked; kiểm tra riêng syntax/import/whitespace các file mới.

Streamlit thực trên `127.0.0.1:8526`, `/_stcore/health` → `ok`. Microsoft Edge headless + Playwright đã đi hết Guided 14 bước, prediction/reveal, seed, BFS partial/final, PCA/rectangle, A/B NFA và final accepted segments. Lab kiểm tra manual seed, noisy preset, τ đổi/reset, candidate B, ε đổi từ REJECT sang ACCEPT, Presentation Mode; quay lại cả năm module cũ. Viewport: desktop **1920×1080**, laptop **1366×768**, tablet **768×1024**, mobile **390×844**. Đã chụp và **mở xem trực quan** ảnh; ở lượt cuối không có horizontal overflow, Streamlit exception, browser console/page error. Ảnh mobile chụp viewport thực tại vị trí cuộn tương ứng, không ghép ảnh. Bài test số học và Manim comparison xác minh phép tính; screenshot xác minh bố cục/trạng thái, không thay cho chứng minh số học.

Ảnh đại diện: [Guided đầu](docs/screenshots/lsd/desktop-guided-initial.png), [gradient](docs/screenshots/lsd/desktop-gradient.png), [seed](docs/screenshots/lsd/desktop-seed.png), [BFS partial](docs/screenshots/lsd/desktop-partial-region.png), [LSR full](docs/screenshots/lsd/desktop-full-region.png), [PCA](docs/screenshots/lsd/desktop-pca-segment.png), [rectangle](docs/screenshots/lsd/desktop-support-rectangle.png), [A trước reveal](docs/screenshots/lsd/desktop-candidate-a-before.png), [A NFA](docs/screenshots/lsd/desktop-candidate-a-nfa.png), [B trước reveal](docs/screenshots/lsd/desktop-candidate-b-before.png), [B NFA](docs/screenshots/lsd/desktop-candidate-b-nfa.png), [all candidates](docs/screenshots/lsd/desktop-all-candidates.png), [Lab manual](docs/screenshots/lsd/desktop-lab-manual-seed.png), [Lab ε đổi](docs/screenshots/lsd/desktop-lab-epsilon-change.png), [Presentation NFA](docs/screenshots/lsd/desktop-presentation-nfa.png), [Laptop Presentation NFA](docs/screenshots/lsd/laptop-presentation-nfa.png), [Tablet Guided](docs/screenshots/lsd/tablet-guided-initial.png), [Mobile Guided](docs/screenshots/lsd/mobile-guided-initial.png), [Mobile Lab](docs/screenshots/lsd/mobile-lab.png), [Mobile Presentation NFA](docs/screenshots/lsd/mobile-presentation-nfa.png).

## Lỗi thấy trong audit và cách sửa

- Gate replay/decision và manual-seed invalidation được siết qua AppTest và browser run.
- Candidate segment ban đầu xanh/cam trước khi reveal, làm lộ decision; chuyển sang vàng trung tính và đổi heading thành “kiểm định”.
- Lab controls đầu tiên đẩy hình xuống sâu trên mobile; thu gọn settings và đặt chart lên trước pixel/replay controls.
- Presentation NFA ban đầu có hình và score tách xa nhau; đặt hai cột ở desktop, tăng metric NFA. Ở laptop 1366px, công thức một dòng bị cắt trong cột; tách thành hai dòng và rút ngắn nhãn `n/k/n−k`, ảnh retest không còn cắt.
- Browser selectbox của Streamlit có thể chỉ focus khi click label, và keyboard slider chỉ commit khi blur; QA dùng Open control/Tab rồi đợi rerun, không đổi thuật toán để che lỗi kiểm thử.

## Giới hạn đã biết

Đây là prototype trên field nhỏ tổng hợp, không phải detector ảnh thật hoặc LSD paper đầy đủ. `N_T` không đại diện toàn bộ rectangle test, vùng chọn thích nghi, nên NFA không hiệu chuẩn guarantee toàn ảnh. Strong gradient có thể từ noise/texture; τ/M/ε đều nhạy với dữ liệu. Axial growth có thể gộp polarity trái ngược nhưng directed k giảm; global reversal thì tangent cũng đảo. Một-pixel region có finite endpoints trùng nhau theo support thật, không tạo chiều dài giả. Streamlit chạy theo hành động/rerun, không drag trực tiếp hoặc video animation.

## Cleanup và Git

Đã đóng các Edge/Playwright browser instances trong `finally` của QA script. Dừng đúng server PID `15196` của tác vụ ở port `8526`; port không còn listener. Đã kiểm tra target nằm trong workspace rồi xóa riêng `chapter3_webapp/.lsd-audit-tools/` (305 file, **40,778,161 file bytes**, gồm package Playwright Core, npm cache, profile/script audit) và 7 screenshot QA trùng trong `chapter3_webapp/docs/screenshots/lsd/` (**675,553 file bytes**); tổng **41,453,714 file bytes**. Free space ổ C đo sát cleanup: **182,343,254,016 → 182,385,184,768 bytes** (tăng **41,930,752 bytes** theo filesystem). Giữ **26** ảnh review và mọi source/tests/docs/config, năm module cũ, Chapter 1, Manim/video. Không xóa cache dùng chung hoặc process không thuộc tác vụ.

`git status --short --branch`: branch `feature/chapter3-manim-animations`; chỉ `?? chapter3_webapp/` vì toàn bộ thư mục Chapter 3 đã untracked từ trước. Không stage/commit/push/merge/rebase.
