"""Feature Matching Challenge: guided application of existing Chapter 2 engines."""

from __future__ import annotations

import json
from dataclasses import replace
from html import escape

import numpy as np
import streamlit as st

from utils.feature_challenge import (Config, METHODS, PRESETS, SCENES,
    build_transformed_image_pair, compute_matching_metrics, detect_pair,
    evaluate_match_geometry, generate_experiment_snapshot, map_points_with_affine,
    match_features, Result)
from utils.feature_challenge_visualization import feature_view, mask_view, pair_view

STEPS = (
    "Bài toán tương ứng", "Pipeline cần gì?", "Chọn phương pháp", "Biến đổi ảnh",
    "Phát hiện keypoint", "Tạo descriptor", "So sánh descriptor",
    "Ứng viên gần nhất", "Loại match mơ hồ", "Kiểm chứng hình học",
    "Đánh giá chất lượng", "So sánh ba pipeline", "Thử một thất bại", "Chọn và bảo vệ phương pháp",
)
QUESTIONS = (
    "Hai ảnh của cùng cảnh: điểm nào thật sự tương ứng?",
    "Vì sao phát hiện góc chưa đủ để ghép ảnh?",
    "Bạn dự đoán phương pháp nào phù hợp với cảnh này?",
    "Biến đổi nào có thể làm descriptor thay đổi?",
    "Phát hiện nhiều điểm có chắc match tốt hơn?",
    "Ba cách mô tả vùng lân cận khác nhau ra sao?",
    "Vì sao Hamming không cùng đơn vị với Euclid?",
    "Láng giềng gần nhất có chắc cùng vị trí vật lý?",
    "Quy tắc descriptor có nhận ứng viên này không?",
    "Ứng viên đã nhận có đúng hình học không?",
    "Precision và coverage được chia cho tập nào?",
    "Cùng ảnh và ngân sách, phương pháp nào có bằng chứng tốt?",
    "Thay một biến có làm giả thuyết của bạn đúng?",
    "Bạn chọn gì cho ứng dụng thực tế, và vì sao?",
)
CHALLENGES = {
    "A · Cặp ảnh sạch": (SCENES[0], "Phương pháp nào tìm tương ứng đáng tin cậy khi biến đổi nhẹ?",
                        "Tăng dịch chuyển, giữ mọi tham số khác.", "Số match không thay thế precision."),
    "B · Quay": (SCENES[3], "Chuẩn hóa hướng ảnh hưởng tới match thế nào?",
                  "Đổi duy nhất góc quay.", "Định hướng không bảo đảm bất biến tuyệt đối."),
    "C · Đổi tỷ lệ": (SCENES[2], "Vì sao patch Harris cố định có thể khó thích nghi?",
                       "Đổi duy nhất scale.", "Ngân sách điểm không đồng nhất descriptor."),
    "D · Hoa văn lặp": (SCENES[1], "Điều gì làm vùng giống nhau trở nên mơ hồ?",
                         "So Nearest với Ratio trên cùng cặp ảnh.", "Ratio/cross-check vẫn có thể nhận match sai."),
    "E · Nhiễu, blur, sáng": (SCENES[5], "Giai đoạn nào dễ nhạy với suy giảm ảnh?",
                             "Đổi duy nhất độ nhiễu.", "Không suy ra xếp hạng phổ quát từ một cảnh."),
}


@st.cache_data(show_spinner=False, max_entries=16)
def _pair_and_detections(scene, angle, scale, dx, dy, brightness, contrast, noise, blur, seed, budget):
    config = Config(scene, angle, scale, dx, dy, brightness, contrast, noise, blur, seed, budget)
    pair = build_transformed_image_pair(config)
    detections = {method: detect_pair(pair, method) for method in METHODS}
    return pair, detections


def _run(config: Config, methods=METHODS) -> list[Result]:
    pair, detections = _pair_and_detections(config.scene, config.angle, config.scale,
        config.dx, config.dy, config.brightness, config.contrast, config.noise,
        config.blur, config.seed, config.budget)
    pair = replace(pair, config=config)
    results = []
    for method in methods:
        a, b = detections[method]
        from time import perf_counter
        start = perf_counter()
        decisions = match_features(a, b, config.rule, config.ratio)
        seconds = perf_counter()-start
        decisions = evaluate_match_geometry(pair, a, b, decisions, config.tolerance)
        results.append(Result(method, pair, a, b, decisions, config.rule, seconds))
    return results


def _state():
    s = st.session_state
    s.setdefault("feature_challenge_step", 1)
    s.setdefault("feature_challenge_mode", s.get("feature_challenge_saved_mode", "Guided Tutorial"))
    s.setdefault("feature_challenge_presentation", s.get("feature_challenge_saved_presentation", False))


def _save_view():
    st.session_state.feature_challenge_saved_mode = st.session_state.feature_challenge_mode
    st.session_state.feature_challenge_saved_presentation = st.session_state.feature_challenge_presentation


def _reset(prefix="feature_challenge_"):
    preserve = {"feature_challenge_mode", "feature_challenge_presentation",
                "feature_challenge_saved_mode", "feature_challenge_saved_presentation"}
    for key in list(st.session_state):
        if key.startswith(prefix) and key not in preserve:
            del st.session_state[key]
    st.session_state.feature_challenge_step = 1


def _reset_workspace(prefix):
    for key in list(st.session_state):
        if key.startswith(prefix):
            del st.session_state[key]


def _move(delta):
    st.session_state.feature_challenge_step = max(1, min(14, st.session_state.feature_challenge_step+delta))


def _plot(fig):
    n = st.session_state.get("feature_challenge_plot_index", 0)
    st.session_state.feature_challenge_plot_index = n+1
    st.plotly_chart(fig, key=f"feature_challenge_plot_{n}", use_container_width=True,
                    config={"displayModeBar": False})


def _load_preset(prefix):
    for suffix in ("angle", "scale", "dx", "dy", "brightness", "contrast", "noise", "blur"):
        st.session_state.pop(prefix+suffix, None)


def _controls(prefix, *, compact=False, budget_control=True) -> Config:
    scene = st.selectbox("Cảnh thực nghiệm", SCENES, key=prefix+"scene",
                         on_change=_load_preset, args=(prefix,))
    defaults = PRESETS[scene]
    target = st.expander("Biến đổi ảnh A → B · hình học và quang học", expanded=not compact)
    with target:
        left, mid, right = st.columns(3)
        with left:
            angle = st.slider("Góc quay (°)", -45., 45., defaults["angle"], 1., key=prefix+"angle")
            scale = st.slider("Scale", .55, 1.4, defaults["scale"], .01, key=prefix+"scale")
            dx = st.slider("Dịch x (px)", -20., 20., defaults["dx"], 1., key=prefix+"dx")
        with mid:
            dy = st.slider("Dịch y (px)", -20., 20., defaults["dy"], 1., key=prefix+"dy")
            brightness = st.slider("Brightness offset", -.25, .25, defaults["brightness"], .01,
                                   key=prefix+"brightness")
            contrast = st.slider("Contrast ×", .4, 1.5, defaults["contrast"], .01,
                                 key=prefix+"contrast")
        with right:
            noise = st.slider("Gaussian noise σ", 0., .15, defaults["noise"], .005,
                              key=prefix+"noise")
            blur = st.slider("Gaussian blur σ", 0., 2.5, defaults["blur"], .1,
                             key=prefix+"blur")
            seed = st.number_input("Seed tái lập", min_value=0, max_value=999999,
                                   value=2026, step=1, key=prefix+"seed")
        st.caption("Góc quay là quy ước OpenCV; ma trận T ánh xạ tâm pixel A → B. "
                   "Brightness/contrast/noise/blur chỉ đổi giá trị intensity.")
    budget = (st.slider("Ngân sách vị trí / phương pháp", 5, 60, 30, 5, key=prefix+"budget")
              if budget_control else 30)
    return Config(scene, angle, scale, dx, dy, brightness, contrast, noise, blur,
                  int(seed), budget)


def _matching_controls(prefix, config):
    a, b, c = st.columns(3)
    rule = a.selectbox("Quy tắc matching", ("Method default", "Nearest", "Ratio", "Cross-check"),
                       key=prefix+"rule")
    ratio = b.slider("Ratio threshold", .5, .95, .8, .01, key=prefix+"ratio",
                     help="Chỉ dùng khi rule=Ratio hoặc Method default cho Harris/SIFT")
    tolerance = c.slider("Sai số hình học tối đa (px)", 1., 10., 4., .5,
                         key=prefix+"tolerance")
    st.caption("Method default: Harris/SIFT dùng Ratio; ORB-style dùng Cross-check. "
               "Muốn so quy tắc tương thích, chọn cùng một rule. Euclid và Hamming không chung đơn vị.")
    return replace(config, rule=rule, ratio=ratio, tolerance=tolerance)


def _pair_panel(pair, *, presentation=False):
    _plot(pair_view(pair, presentation=presentation))
    st.caption("Mọi phương pháp nhận cùng một cặp ảnh. Vệt tối/đệm sau warp không là dữ liệu quan sát hợp lệ.")


def _comparison_rows(results):
    """One computed view model feeds both desktop table and mobile cards."""
    rows = []
    for r in results:
        m = compute_matching_metrics(r)
        rows.append({"Pipeline":r.method,
                     "Rule":("Cross-check" if r.rule=="Method default" and r.method==METHODS[2]
                             else "Ratio" if r.rule=="Method default" else r.rule),
                     "Có trước cap":m["available_a"],
                     "Vị trí giữ":m["unique_a"], "Bản ghi hướng":m["records_a"],
                     "Descriptor hợp lệ":m["descriptors_a"], "Ứng viên":m["candidates"],
                     "Nhận":m["accepted"], "Đúng hình học":m["correct"],
                     "Precision":f'{m["precision"]:.1%}' if m["precision"] is not None else "N/A",
                     "Coverage":f'{m["coverage"]:.1%}' if m["coverage"] is not None else "N/A",
                     "Median lỗi (px)":f'{m["median_error"]:.2f}' if m["median_error"] is not None else "N/A",
                     "ms · teaching":f'{1000*(m["detection_seconds"]+m["matching_seconds"]):.0f}'})
    return rows


def _metric_table(results, reveal=True):
    if not reveal:
        st.info("Hãy ghi dự đoán rồi Reveal để xem số liệu hình học.")
        return
    rows = _comparison_rows(results)
    if len(rows)>1:
        rules={row["Rule"] for row in rows}
        st.warning("Chính sách matching trong bảng: " +
                   "; ".join(f'{row["Pipeline"]}: {row["Rule"]}' for row in rows) +
                   (". Đây là các rule khác nhau; chọn cùng Ratio/Cross-check/Nearest ở trên rồi chạy lại để so theo một chính sách."
                    if len(rules)>1 else ". Cùng rule, nhưng Euclid và Hamming vẫn không cùng đơn vị."))
    with st.container(key="feature_comparison_desktop"):
        st.dataframe(rows, hide_index=True, width="stretch")
    with st.container(key="feature_comparison_mobile"):
        for row in rows:
            st.markdown(
                f'<div class="feature-result-card"><strong>{escape(row["Pipeline"])}</strong>'
                f'<small>Rule: {escape(row["Rule"])}</small>'
                f'<div>Đúng / nhận: {row["Đúng hình học"]} / {row["Nhận"]}</div>'
                f'<div>Precision: {row["Precision"]} · Coverage: {row["Coverage"]}</div>'
                f'<div>Descriptor hợp lệ: {row["Descriptor hợp lệ"]} · '
                f'Thời gian teaching: {row["ms · teaching"]} ms</div></div>',
                unsafe_allow_html=True)
    st.caption("Precision = đúng / match đã nhận và đánh giá được; 0 match → N/A. "
               "Coverage = vị trí A có ≥1 match đúng / vị trí A có descriptor hợp lệ, quan sát được; "
               "không phải detector recall. Median tính trên mọi match nhận đánh giá được. "
               "Thời gian là lần tính educational engine gốc (có thể được lấy lại từ cache), "
               "không phải thời gian bấm nút hoặc benchmark production.")


def _gate(key, question, options, signature):
    sigkey = key+"_signature"
    if st.session_state.get(sigkey) != signature:
        st.session_state[sigkey] = signature
        st.session_state.pop(key+"_guess", None)
        st.session_state.pop(key+"_reveal", None)
    st.markdown(f'<div class="question">DỰ ĐOÁN · {escape(question)}</div>', unsafe_allow_html=True)
    guess = st.radio("Chọn trước khi Reveal", options, index=None, key=key+"_guess", horizontal=True)
    if st.button("Reveal kết quả", key=key+"_button", disabled=guess is None):
        st.session_state[key+"_reveal"] = True
    return bool(st.session_state.get(key+"_reveal", False)), guess


def _nav(allowed=True):
    step = st.session_state.feature_challenge_step
    a, b, c = st.columns(3)
    a.button("← Previous", on_click=_move, args=(-1,), disabled=step == 1, width="stretch")
    b.button("Next →", on_click=_move, args=(1,), disabled=step == 14 or not allowed, width="stretch")
    c.button("Reset", on_click=_reset, width="stretch")


def _selected_match(result, prefix, *, reveal_rule=False, reveal_geometry=False, presentation=False):
    candidates = [i for i, d in enumerate(result.decisions) if d.train is not None]
    if not candidates:
        _plot(pair_view(result.pair, presentation=presentation))
        st.warning("Không có ứng viên có descriptor hợp lệ ở cả hai ảnh. Hãy đổi cảnh hoặc ngân sách.")
        return None
    index = st.selectbox("Ứng viên A → B", candidates, key=prefix+"candidate",
                         format_func=lambda i:f"#{i+1} · A[{result.decisions[i].query}] → B[{result.decisions[i].train}]")
    d = result.decisions[index]
    options=["Tất cả ứng viên"]
    if reveal_rule:options.append("Rule đã nhận")
    if reveal_geometry:options.append("Đúng/sai hình học")
    filter_mode=st.selectbox("Lọc đường match",options,
                             key=prefix+f"filter_{int(reveal_rule)}_{int(reveal_geometry)}")
    _plot(pair_view(result.pair, result, filter_mode=filter_mode,selected=index, reveal_rule=reveal_rule,
                    reveal_geometry=reveal_geometry, presentation=presentation))
    st.caption("Đường được vẽ tối đa 24 ứng viên đầu để dễ đọc; bảng metrics tính toàn bộ. "
               "Xám xanh = chưa kiểm chứng; xám = bị rule loại; xanh = đúng hình học; cam = sai hình học.") if reveal_geometry else st.caption(
                   "Đường màu trung tính: chưa tiết lộ kết quả hình học. Tối đa 24 đường đầu hiển thị; metrics tính toàn bộ.")
    a = result.a.features[d.query]
    b = result.b.features[d.train]
    st.write(f"A=({a.x:.1f}, {a.y:.1f}) → B=({b.x:.1f}, {b.y:.1f}); "
             f"d₁={d.distance:.3f}, d₂={d.second_distance:.3f}" if d.second_distance is not None else
             f"A=({a.x:.1f}, {a.y:.1f}) → B=({b.x:.1f}, {b.y:.1f}); d₁={d.distance:.3f}, d₂=N/A")
    with st.expander("Descriptor truy vấn và hai láng giềng"):
        second = result.b.features[d.second] if d.second is not None else None
        st.write(f"Láng giềng #1: B[{d.train}] ({b.x:.1f},{b.y:.1f}), d₁={d.distance:.3f}.")
        st.write(f"Láng giềng #2: B[{d.second}] ({second.x:.1f},{second.y:.1f}), d₂={d.second_distance:.3f}."
                 if second is not None else "Láng giềng #2: N/A; ratio test không thể quyết định.")
        st.write(f"Metadata detector: {a.metadata}")
        st.code(" ".join(f"{byte:02x}" for byte in a.descriptor[:16]) + " …"
                if isinstance(a.descriptor,bytes) else np.array2string(a.descriptor[:16],precision=3),
                language="text")
    if reveal_rule:
        st.write(f"Quyết định descriptor: {'NHẬN' if d.accepted else 'LOẠI'} · {d.reason}.")
    if reveal_geometry and d.accepted:
        expected = map_points_with_affine([[a.x,a.y]], result.pair.matrix)[0]
        st.write(f"T(A)=({expected[0]:.2f}, {expected[1]:.2f}); lỗi={d.error:.2f} px; "
                 f"{'ĐÚNG' if d.correct else 'SAI'} tại ngưỡng {result.pair.config.tolerance:g} px.")
    return d


def _guided():
    s = st.session_state
    step = s.feature_challenge_step
    st.markdown(f'<div class="question">{escape(QUESTIONS[step-1])}</div>', unsafe_allow_html=True)
    config = _controls("feature_challenge_g_", compact=True, budget_control=step>=5)
    config = _matching_controls("feature_challenge_g_", config) if step >= 8 else config
    signature = (config.scene, config.angle, config.scale, config.dx, config.dy,
                 config.brightness, config.contrast, config.noise, config.blur,
                 config.seed, config.budget, config.rule, config.ratio)
    if s.get("feature_challenge_g_signature") != signature:
        s.feature_challenge_g_signature = signature
        for n in (3, 9, 10, 12, 13, 14):
            s.pop(f"feature_challenge_g_gate_{n}_reveal", None)
            s.pop(f"feature_challenge_g_gate_{n}_guess", None)
        s.pop("feature_challenge_g_final_reveal", None)
    allowed = True
    if step <= 4:
        pair = build_transformed_image_pair(config)
        _pair_panel(pair, presentation=s.feature_challenge_presentation)
        if step == 1:
            st.info("Một điểm A chỉ trở thành correspondence khi descriptor chọn điểm B và vị trí B phù hợp hình học.")
        elif step == 2:
            st.write("Detection → Description → Matching → Verification. Harris chỉ phát hiện; patch 11×11 là baseline bổ sung.")
        elif step == 3:
            revealed, guess = _gate("feature_challenge_g_gate_3", "Phương pháp nào bạn chọn trước khi đo?", METHODS, signature)
            allowed = revealed
            if revealed: st.success(f"Đã khóa dự đoán: {guess}. Chưa có phương pháp thắng mặc định; hãy đo ở bước sau.")
        else:
            st.code(np.array2string(pair.matrix, precision=3), language="text")
            st.write("T là phép ánh xạ forward A → B. warpAffine dùng inverse sampling nội bộ; tâm pixel giữ quy ước OpenCV.")
    else:
        selected_method = st.selectbox("Pipeline cần soi", METHODS, key="feature_challenge_g_method")
        results = _run(config)
        result = next(r for r in results if r.method == selected_method)
        metrics = compute_matching_metrics(result)
        if step == 5:
            a,b = st.columns(2)
            with a: _plot(feature_view(result,"A",s.feature_challenge_presentation))
            with b: _plot(feature_view(result,"B",s.feature_challenge_presentation))
            st.write(f"A: {metrics['available_a']} vị trí trước cap → {metrics['unique_a']} vị trí giữ; "
                     f"{metrics['records_a']} bản ghi hướng. SIFT có thể có nhiều hướng tại một vị trí.")
        elif step == 6:
            _plot(feature_view(result,"A",s.feature_challenge_presentation))
            feature = next((f for f in result.a.features if f.descriptor is not None), None)
            if feature is None: st.warning("Không có descriptor hợp lệ sau kiểm tra vùng support.")
            else:
                d = feature.descriptor
                st.write(f"{selected_method}: payload {metrics['descriptor_bytes']} byte; metadata: {feature.metadata}")
                st.code((" ".join(f"{x:02x}" for x in d[:16]) + " …") if isinstance(d,bytes)
                        else np.array2string(d[:16],precision=2), language="text")
                st.caption("Harris: trừ mean rồi chuẩn L2, patch hằng → vector zero. SIFT 128D; ORB-style 256 bit / 32 byte.")
        elif step == 7:
            _plot(pair_view(result.pair, result, presentation=s.feature_challenge_presentation))
            st.write("Harris/SIFT: Euclid trên vector float; ORB-style: XOR + popcount trên 256 bit. Không so số d của hai hệ trực tiếp.")
        elif step == 8:
            _selected_match(result,"feature_challenge_g8_",presentation=s.feature_challenge_presentation)
            st.info("Láng giềng thứ nhất là ứng viên theo descriptor, chưa phải một sự thật hình học.")
        elif step == 9:
            d = _selected_match(result,"feature_challenge_g9_",presentation=s.feature_challenge_presentation)
            if d:
                revealed, _ = _gate("feature_challenge_g_gate_9", "Rule sẽ NHẬN ứng viên này?", ("Có","Không"), signature+(d.query,))
                allowed = revealed
                if revealed: st.write(f"Rule: {'NHẬN' if d.accepted else 'LOẠI'} · {d.reason}.")
        elif step == 10:
            d = _selected_match(result,"feature_challenge_g10_",reveal_rule=True,
                                presentation=s.feature_challenge_presentation)
            if d:
                revealed, _ = _gate("feature_challenge_g_gate_10", "Match đã nhận có đúng hình học?", ("Có","Không"), signature+(d.query,config.tolerance))
                allowed = revealed
                if revealed:
                    index = next(i for i, item in enumerate(result.decisions) if item.query == d.query)
                    _plot(pair_view(result.pair, result, selected=index, reveal_rule=True,
                                    reveal_geometry=True, presentation=s.feature_challenge_presentation))
                    source = result.a.features[d.query]
                    expected = map_points_with_affine([[source.x,source.y]],result.pair.matrix)[0]
                    st.write(f"T(A)=({expected[0]:.2f},{expected[1]:.2f}); lỗi={d.error:.2f}px; "
                             f"{'ĐÚNG' if d.correct else 'SAI'} ở ngưỡng {config.tolerance:g}px.")
        elif step == 11:
            _plot(pair_view(result.pair,result,reveal_rule=True,reveal_geometry=True,
                            presentation=s.feature_challenge_presentation))
            _metric_table([result])
        elif step == 12:
            _plot(pair_view(result.pair,result,presentation=s.feature_challenge_presentation))
            revealed,_ = _gate("feature_challenge_g_gate_12", "Theo precision, bạn dự đoán pipeline nào cao nhất?", METHODS, signature)
            allowed = revealed
            _metric_table(results,revealed)
        elif step == 13:
            st.info("Hãy chọn cảnh hoa văn lặp/đổi tỷ lệ/nhiễu, ghi giả thuyết và thay đúng một biến ở trên.")
            _plot(pair_view(result.pair,result,presentation=s.feature_challenge_presentation))
            revealed,_ = _gate("feature_challenge_g_gate_13", "Tăng biến đổi sẽ làm precision giảm?", ("Có thể","Không chắc"), signature)
            allowed = revealed
            _metric_table(results,revealed)
        else:
            _plot(pair_view(result.pair,result,presentation=s.feature_challenge_presentation))
            choice = st.selectbox("Chọn cho ứng dụng",METHODS,key="feature_challenge_g_choice")
            reason = st.text_area("Giải thích bằng ít nhất một số liệu đo được",key="feature_challenge_g_reason")
            if st.button("Ghi nhận lập luận",disabled=len(reason.strip())<12):
                s.feature_challenge_g_final_reveal = True
            if s.get("feature_challenge_g_final_reveal"):
                _metric_table(results)
                st.success(f"Đã chọn {choice}. Hãy cân đối scene, biến đổi, precision, coverage, payload và giới hạn của teaching engines.")
            else: allowed=False
    _nav(allowed)


def _lab_a():
    st.subheader("Lab A · Image Transformation Studio")
    config = _controls("feature_challenge_a_")
    pair = build_transformed_image_pair(config)
    _pair_panel(pair)
    col, mask = st.columns(2)
    with col:
        point = st.slider("Soi tâm pixel nguồn x", 10, 118, 64, key="feature_challenge_a_x")
        mapped = map_points_with_affine([[point,64]],pair.matrix)[0]
        st.code(np.array2string(pair.matrix,precision=3),language="text")
        st.write(f"A=({point},64) → T(A)=({mapped[0]:.2f},{mapped[1]:.2f}) trên B.")
        st.caption("T forward; warpAffine nội bộ lấy mẫu qua T⁻¹.")
    with mask:
        _plot(mask_view(pair))
        st.caption("Xanh: pixel B có support nguồn đầy đủ; cam: vùng đệm/partial support.")
    st.button("Reset Lab A",on_click=_reset_workspace,args=("feature_challenge_a_",))


def _lab_b():
    st.subheader("Lab B · Matching Microscope")
    config = _controls("feature_challenge_b_",compact=True)
    config = _matching_controls("feature_challenge_b_",config)
    method = st.selectbox("Pipeline",METHODS,key="feature_challenge_b_method")
    signature=(replace(config,tolerance=4.),method,
               st.session_state.get("feature_challenge_b_candidate"))
    if st.session_state.get("feature_challenge_b_signature")!=signature:
        st.session_state.feature_challenge_b_signature=signature
        st.session_state.pop("feature_challenge_b_rule_reveal",None)
        st.session_state.pop("feature_challenge_b_geom_reveal",None)
    result = _run(config,(method,))[0]
    d = _selected_match(result,"feature_challenge_b_",
                        reveal_rule=bool(st.session_state.get("feature_challenge_b_rule_reveal")),
                        reveal_geometry=bool(st.session_state.get("feature_challenge_b_geom_reveal")))
    if d:
        a,b=st.columns(2)
        guess=a.radio("Rule có nhận?",("Có","Không"),index=None,key="feature_challenge_b_rule_guess")
        if a.button("Reveal quyết định",disabled=guess is None):
            st.session_state.feature_challenge_b_rule_reveal=True
        if st.session_state.get("feature_challenge_b_rule_reveal"):
            st.write(f"{'Nhận' if d.accepted else 'Loại'}: {d.reason}. "
                     "Descriptor không biết ma trận T.")
            geometric=b.radio("Đúng hình học?",("Có","Không"),index=None,
                              key="feature_challenge_b_geom_guess")
            if b.button("Reveal hình học",disabled=geometric is None):
                st.session_state.feature_challenge_b_geom_reveal=True
            if st.session_state.get("feature_challenge_b_geom_reveal"):
                index = next(i for i, item in enumerate(result.decisions) if item.query == d.query)
                _plot(pair_view(result.pair, result, selected=index, reveal_rule=True,reveal_geometry=True))
                if d.accepted:
                    source=result.a.features[d.query]
                    expected=map_points_with_affine([[source.x,source.y]],result.pair.matrix)[0]
                    st.write(f"T(A)=({expected[0]:.2f},{expected[1]:.2f}); lỗi={d.error:.2f}px; "
                             f"{'ĐÚNG' if d.correct else 'SAI'} ở ngưỡng {config.tolerance:g}px.")
                else:
                    st.write("Ứng viên đã bị rule loại; không đưa vào precision hậu kiểm.")
        st.caption("Chọn ứng viên hoặc đổi ảnh/rule → reveal cũ bị vô hiệu. Ngưỡng hình học chỉ đổi phân loại hậu kiểm.")
    st.button("Reset Lab B",on_click=_reset_workspace,args=("feature_challenge_b_",))


def _lab_c():
    st.subheader("Lab C · Algorithm Comparison Arena")
    config = _controls("feature_challenge_c_",compact=True)
    config = _matching_controls("feature_challenge_c_",config)
    methods=st.multiselect("Pipeline đưa vào arena",METHODS,default=list(METHODS),
                           key="feature_challenge_c_methods")
    signature=(replace(config,tolerance=4.),tuple(methods))
    if st.button("Run comparison",disabled=not methods):
        st.session_state.feature_challenge_c_results=_run(config,methods)
        st.session_state.feature_challenge_c_signature=signature
        st.session_state.feature_challenge_c_reveal=False
    results=st.session_state.get("feature_challenge_c_results")
    if results and st.session_state.get("feature_challenge_c_signature")!=signature:
        st.warning("Cấu hình đã đổi. Nhấn Run comparison để tính lại; kết quả cũ không được gán cho ảnh mới.")
        results=None
    elif results and results[0].pair.config.tolerance!=config.tolerance:
        results=[replace(r,pair=replace(r.pair,config=config),
                         decisions=evaluate_match_geometry(r.pair,r.a,r.b,r.decisions,
                                                           config.tolerance)) for r in results]
        st.session_state.feature_challenge_c_results=results
        st.session_state.feature_challenge_c_reveal=False
        st.info("Chỉ đổi ngưỡng hình học: giữ nguyên descriptor và matching decisions; "
                "phân loại đúng/sai đã tính lại. Reveal lại để xem.")
    if results:
        method=st.selectbox("Soi pipeline",methods,key="feature_challenge_c_inspect")
        selected=next(r for r in results if r.method==method)
        if st.button("Reveal geometry metrics"):
            st.session_state.feature_challenge_c_reveal=True
        revealed=bool(st.session_state.get("feature_challenge_c_reveal",False))
        options=["Tất cả ứng viên","Rule đã nhận"]
        if revealed:options.append("Đúng/sai hình học")
        filter_mode=st.selectbox("Lọc đường match",options,
                                 key=f"feature_challenge_c_filter_{int(revealed)}")
        _plot(pair_view(selected.pair,selected,filter_mode=filter_mode,reveal_rule=True,
                        reveal_geometry=revealed))
        _metric_table(results,revealed)
        st.text_input("Dự đoán",key="feature_challenge_c_prediction")
        st.text_area("Giải thích",key="feature_challenge_c_explanation")
        if revealed:
            snapshot=generate_experiment_snapshot(results,
                st.session_state.get("feature_challenge_c_prediction",""),
                st.session_state.get("feature_challenge_c_explanation",""))
            if st.button("Save snapshot"):
                st.session_state.setdefault("feature_challenge_c_snapshots",[]).append(snapshot)
            st.download_button("Export JSON kết quả",json.dumps(snapshot,ensure_ascii=False,indent=2).encode("utf-8"),
                               file_name="feature_challenge.json",mime="application/json")
    saved=st.session_state.get("feature_challenge_c_snapshots",[])
    if saved:
        st.write(f"Đã lưu {len(saved)} snapshot trong session này (không lưu bền vững).")
        st.dataframe([{"ID":s["id"],"Prediction":s["prediction"],"Scene":s["config"]["scene"],
                       "Methods":len(s["metrics"])} for s in saved],hide_index=True,width="stretch")
    st.button("Reset Lab C",on_click=_reset_workspace,args=("feature_challenge_c_",))


def _challenge():
    st.subheader("Classroom Challenge · dự đoán trước khi đo")
    name=st.selectbox("Kịch bản",tuple(CHALLENGES),key="feature_challenge_class_scenario")
    scene,question,followup,limitation=CHALLENGES[name]
    config=Config(scene=scene,**PRESETS[scene])
    signature=(name,config)
    if st.session_state.get("feature_challenge_class_signature")!=signature:
        st.session_state.feature_challenge_class_signature=signature
        for key in ("prediction","locked","results","revealed","explanation"):
            st.session_state.pop("feature_challenge_class_"+key,None)
    st.markdown(f'<div class="question">{escape(question)}</div>',unsafe_allow_html=True)
    _pair_panel(build_transformed_image_pair(config),presentation=st.session_state.feature_challenge_presentation)
    prediction=st.radio("Dự đoán theo precision",METHODS,index=None,
                        key="feature_challenge_class_prediction",
                        disabled=st.session_state.get("feature_challenge_class_locked",False))
    if st.button("Lock prediction",disabled=prediction is None or st.session_state.get("feature_challenge_class_locked",False)):
        st.session_state.feature_challenge_class_locked=True
    if st.session_state.get("feature_challenge_class_locked"):
        if st.button("Run scenario"):
            st.session_state.feature_challenge_class_results=_run(config)
        results=st.session_state.get("feature_challenge_class_results")
        if results:
            if st.button("Reveal measured quality"):
                st.session_state.feature_challenge_class_revealed=True
            revealed=st.session_state.get("feature_challenge_class_revealed",False)
            _metric_table(results,revealed)
            if revealed:
                selected=st.selectbox("Soi pipeline và failure",METHODS,key="feature_challenge_class_inspect")
                result=next(r for r in results if r.method==selected)
                correct=next((i for i,d in enumerate(result.decisions) if d.accepted and d.correct),None)
                wrong=next((i for i,d in enumerate(result.decisions) if d.accepted and d.correct is False),None)
                st.write(f"Ví dụ đúng: {'#'+str(correct+1) if correct is not None else 'không có'}; "
                         f"ví dụ sai: {'#'+str(wrong+1) if wrong is not None else 'không có'}.")
                _plot(pair_view(result.pair,result,selected=wrong if wrong is not None else correct,
                                reveal_rule=True,reveal_geometry=True,
                                presentation=st.session_state.feature_challenge_presentation))
                explanation=st.text_area("Giải thích một kết quả hoặc thất bại",key="feature_challenge_class_explanation")
                if len(explanation.strip())>=12:
                    st.info(f"Thử tiếp: {followup} Giới hạn cần nêu: {limitation}")
    st.button("Reset Challenge",on_click=_reset_workspace,args=("feature_challenge_class_",))


def render():
    _state()
    st.session_state.feature_challenge_plot_index=0
    st.markdown('<div class="chapter-kicker">CHAPTER 2 · APPLIED INTEGRATION LAB</div>',unsafe_allow_html=True)
    st.title("Feature Matching Challenge")
    st.markdown('<p class="feature-challenge-intro">Chọn → Dự đoán → Cấu hình → Chạy → Soi → Đánh giá → Giải thích. '
                'Module ứng dụng tổng hợp, không phải thuật toán thứ năm trong slide.</p>',
                unsafe_allow_html=True)
    a,b=st.columns([4,1])
    with a:
        mode=st.radio("Chế độ",("Guided Tutorial","Experiment Lab","Classroom Challenge"),
                      horizontal=True,key="feature_challenge_mode",on_change=_save_view)
    with b:
        st.toggle("Presentation Mode",key="feature_challenge_presentation",on_change=_save_view)
    if mode=="Guided Tutorial":
        step=st.session_state.feature_challenge_step
        st.progress(step/14,text=f"{step}/14 · {STEPS[step-1]}")
        _guided()
    elif mode=="Experiment Lab":
        tab_a,tab_b,tab_c=st.tabs(("A · Transform","B · Microscope","C · Compare"))
        with tab_a:_lab_a()
        with tab_b:_lab_b()
        with tab_c:_lab_c()
    else:
        _challenge()
    st.caption("Teaching implementations; descriptor payload excludes metadata/cache. "
               "ORB-style uses a deterministic educational pattern, not OpenCV's learned ORB pattern.")
