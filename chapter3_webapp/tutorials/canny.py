"""One-image, inspectable Canny tutorial and experiment workspaces."""

from __future__ import annotations

import hashlib
from html import escape

import numpy as np
import streamlit as st

from utils.canny import (PRESETS, base_stages, inspect_nms, nms_examples, preset_image,
                         replay_frame, threshold_stages, weak_examples)
from utils.canny_visualization import STAGES, image_figure, kernel_figure
from utils.visualization import PLOT_CONFIG

STEPS = (
    "Vì sao tìm biên?", "Một ảnh, nhiều giai đoạn", "Gaussian kernel", "Làm mượt",
    "Sobel gradient", "Magnitude & hướng", "Vì sao biên dày?", "NMS · KEEP",
    "NMS · SUPPRESS", "NMS toàn ảnh", "Double threshold", "Hysteresis · lan truyền",
    "Weak cô lập", "Edge map & giới hạn",
)
QUESTIONS = {
    8: ("Pixel p có phải cực đại dọc hướng gradient?", ("KEEP", "SUPPRESS", "Chưa chắc")),
    9: ("Pixel p này sẽ được giữ lại?", ("KEEP", "SUPPRESS", "Chưa chắc")),
    11: ("Có nên bỏ mọi pixel weak ngay lập tức?", ("Có", "Không", "Chưa chắc")),
    14: ("Canny luôn trả đúng biên vật thể?", ("Có", "Không", "Chưa chắc")),
}
PRESET_QUESTIONS = {
    PRESETS[0]: "Biên hình học có tách khỏi texture/noise sau toàn pipeline?",
    PRESETS[1]: "Biên sạch có ổn định khi đổi σ và threshold?",
    PRESETS[2]: "Tăng σ giảm noise nhưng có làm mất chi tiết?",
    PRESETS[3]: "Đoạn weak nối với strong có được giữ qua hysteresis?",
    PRESETS[4]: "Weak tách khỏi strong có bị loại?",
    PRESETS[5]: "Làm mượt quá mạnh ảnh hưởng hai nét mảnh thế nào?",
}
PRESET_TAKEAWAYS = {
    PRESETS[0]: "Biên hình học nổi bật, nhưng texture và noise vẫn có thể tạo response; ngưỡng không phải hiểu biết ngữ nghĩa.",
    PRESETS[1]: "Biên sạch thường ổn định, song blur lớn và threshold cao vẫn có thể làm mất nét.",
    PRESETS[2]: "Gaussian giảm đáp ứng noise; σ lớn quá làm nhòe cả chi tiết thật.",
    PRESETS[3]: "Đoạn weak nối với strong được giữ theo 8-connectivity, không bị đổi thành strong.",
    PRESETS[4]: "Weak không có đường nối tới strong bị loại, dù NMS response khác 0.",
    PRESETS[5]: "Các nét gần nhau có thể hòa vào nhau khi σ tăng; không có σ tốt cho mọi kích thước chi tiết.",
}


@st.cache_data(show_spinner=False, max_entries=18)
def _base_cached(image_bytes: bytes, shape: tuple[int, int], sigma: float) -> dict:
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    return base_stages(image, sigma)


@st.cache_data(show_spinner=False, max_entries=24)
def _threshold_cached(image_bytes: bytes, shape: tuple[int, int], sigma: float,
                      low: float, high: float) -> dict:
    return threshold_stages(_base_cached(image_bytes, shape, sigma), low, high)


def _init() -> None:
    state = st.session_state
    state.setdefault("canny_guided_step", 1)
    state.setdefault("canny_preset", PRESETS[0])
    state.setdefault("canny_sigma", 1.15)
    state.setdefault("canny_low", 0.10)
    state.setdefault("canny_high", 0.35)
    state.setdefault("canny_point", (130, 116))
    state.setdefault("canny_bfs_step", 0)
    state.setdefault("canny_lab_stage", "Original")
    state.setdefault("canny_workspace", "A · Pipeline Explorer")


def _invalidate() -> None:
    state = st.session_state
    state.canny_bfs_step = 0
    for step in QUESTIONS:
        state.pop(f"canny_predict_{step}", None)
        state.pop(f"canny_reveal_{step}", None)
    state.pop("canny_nms_reveal", None)
    state.pop("canny_nms_predict", None)
    state.pop("canny_preset_predict", None)
    state.pop("canny_preset_reveal", None)


def _load() -> dict | None:
    _init()
    state = st.session_state
    image = preset_image(state.canny_preset)
    low, high = float(state.canny_low), float(state.canny_high)
    sigma = float(state.canny_sigma)
    signature = (hashlib.sha256(image.tobytes()).hexdigest(), sigma, low, high)
    if state.get("canny_signature") != signature:
        _invalidate()
        state.canny_signature = signature
    row, col = state.canny_point
    if not (1 <= row < image.shape[0] - 1 and 1 <= col < image.shape[1] - 1):
        state.canny_point = (image.shape[0] // 2, image.shape[1] // 2)
    if not (0 < low < high <= 1):
        st.error("Ngưỡng không hợp lệ: cần 0 < low < high ≤ 1. Hãy sửa giá trị; không tự đổi thứ tự.")
        return None
    return _threshold_cached(image.tobytes(), image.shape, sigma, low, high)


def _header(title: str, purpose: str) -> None:
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("canny_presentation", False):
        st.markdown(f'<div class="hero-copy">{purpose}</div>', unsafe_allow_html=True)


def _progress(step: int) -> None:
    cells = "".join(f'<div class="step {"active" if i == step else "done" if i < step else ""}">{i}<br>{escape(name)}</div>'
                    for i, name in enumerate(STEPS, 1))
    st.markdown(f'<div class="stepper canny-stepper">{cells}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-progress">Tiến độ học · {step}/14'
                f'<div class="progress-track"><span style="width:{step/14*100:.1f}%"></span></div></div>',
                unsafe_allow_html=True)


def _guided_move(delta: int) -> None:
    st.session_state.canny_guided_step = min(14, max(1, st.session_state.canny_guided_step + delta))


def _guided_reset() -> None:
    state = st.session_state
    state.canny_guided_step = 1
    state.canny_preset = PRESETS[0]
    state.canny_sigma = 1.15
    state.canny_low = .10
    state.canny_high = .35
    state.canny_point = (130, 116)
    _invalidate()


def _navigation(step: int) -> None:
    with st.container(key="canny_nav"):
        a, b, c = st.columns(3)
        a.button("← Previous", key="canny_previous", disabled=step == 1, width="stretch",
                 on_click=_guided_move, args=(-1,))
        gated = step in (8, 9, 11) and not st.session_state.get(f"canny_reveal_{step}", False)
        b.button("Next →", key="canny_next", disabled=step == 14 or gated, width="stretch",
                 on_click=_guided_move, args=(1,))
        c.button("Reset", key="canny_reset", width="stretch", on_click=_guided_reset)


def _prediction(step: int, answer: str) -> None:
    question, options = QUESTIONS[step]
    st.markdown(f'<div class="question">Dự đoán: {escape(question)}</div>', unsafe_allow_html=True)
    key = f"canny_predict_{step}"
    st.radio("Chọn trước khi reveal", options, index=None, key=key)
    st.button("Reveal Prediction", key=f"canny_reveal_button_{step}",
              disabled=not bool(st.session_state.get(key)),
              on_click=lambda: st.session_state.__setitem__(f"canny_reveal_{step}", True))
    if st.session_state.get(f"canny_reveal_{step}"):
        st.success(answer)


def _controls(*, lab: bool = False) -> None:
    with st.expander("Ảnh và tham số pipeline" if not lab else "Chọn ảnh và tham số", expanded=False):
        st.selectbox("Teaching preset", PRESETS, key="canny_preset", disabled=not lab)
        a, b, c = st.columns(3)
        a.slider("Gaussian σ", .45, 3.5, step=.05, key="canny_sigma")
        b.number_input("Low threshold", min_value=0.0, max_value=1.0, step=.01,
                       format="%.2f", key="canny_low")
        c.number_input("High threshold", min_value=0.0, max_value=1.0, step=.01,
                       format="%.2f", key="canny_high")
        if lab:
            st.button("Reset Experiment", on_click=_reset_lab)
        st.caption("σ đổi → Smooth, Sobel, NMS và edge map đều tính lại. Threshold áp trên NMS đã chuẩn hóa lần hai.")


def _reset_lab() -> None:
    state = st.session_state
    state.canny_preset = PRESETS[0]
    state.canny_sigma = 1.15
    state.canny_low = .10
    state.canny_high = .35
    state.canny_point = (130, 116)
    state.canny_lab_stage = "Original"
    _invalidate()


def _point_controls(arrays: dict, *, key: str) -> tuple[int, int]:
    height, width = arrays["original"].shape
    row, col = st.session_state.canny_point
    state = st.session_state
    if state.get(f"{key}_anchor") != (row, col):
        state[f"{key}_row"] = int(row)
        state[f"{key}_col"] = int(col)
    state[f"{key}_anchor"] = (row, col)
    with st.container(key=key):
        a, b = st.columns(2)
        a.number_input("Row y ↓", 1, height - 2, key=f"{key}_row",
                       on_change=_point_changed, args=(key,))
        b.number_input("Column x →", 1, width - 2, key=f"{key}_col",
                       on_change=_point_changed, args=(key,))
    return st.session_state.canny_point


def _point_changed(key: str) -> None:
    state = st.session_state
    state.canny_point = (int(state[f"{key}_row"]), int(state[f"{key}_col"]))
    state[f"{key}_anchor"] = state.canny_point
    state.canny_nms_reveal = False
    for step in (8, 9):
        state.pop(f"canny_reveal_{step}", None)


def _set_point(point: tuple[int, int]) -> None:
    st.session_state.canny_point = (int(point[0]), int(point[1]))
    st.session_state.canny_nms_reveal = False
    for step in (8, 9):
        st.session_state.pop(f"canny_reveal_{step}", None)


def _nms_inspector(arrays: dict, *, gate: bool = False) -> None:
    options = nms_examples(arrays)
    quick = [("Surviving maximum", "keep"), ("Suppressed shoulder", "suppress"),
             ("0°", "sector_0"), ("45°", "sector_45"), ("90°", "sector_90"), ("135°", "sector_135")]
    cols = st.columns(3)
    for index, (label, name) in enumerate(quick):
        point = options[name]
        cols[index % 3].button(label, key=f"canny_example_{name}", disabled=point is None,
                               on_click=_set_point, args=(point,) if point is not None else (), width="stretch")
    point = _point_controls(arrays, key="canny_nms_point")
    comparison = inspect_nms(arrays["magnitude"], arrays["direction"], point)
    st.plotly_chart(image_figure(arrays, "Magnitude", point=point, comparison=comparison,
                                 presentation=st.session_state.get("canny_presentation", False), patch=True),
                    config=PLOT_CONFIG, key="canny_nms_patch")
    st.caption("7×7 từ đúng tọa độ ảnh gốc · p vàng; q1/q2 xanh. Giá trị dưới đây lấy từ M trước NMS.")
    st.markdown(f"**p** {point} · **q1** {comparison['q1']} · **q2** {comparison['q2']}  \n"
                f"θ thô = {comparison['raw_degrees']:.1f}° → sector **{comparison['degrees']}°**  \n"
                f"M(q1) = **{comparison['m1']:.3f}** · M(p) = **{comparison['magnitude']:.3f}** · "
                f"M(q2) = **{comparison['m2']:.3f}**")
    if gate:
        return
    st.radio("Dự đoán quyết định NMS", ("KEEP", "SUPPRESS"), index=None, key="canny_nms_predict")
    st.button("Reveal NMS decision", disabled=not st.session_state.get("canny_nms_predict"),
              on_click=lambda: st.session_state.__setitem__("canny_nms_reveal", True))
    if st.session_state.get("canny_nms_reveal", False):
        outcome = "KEEP" if comparison["keep"] else "SUPPRESS"
        relation = "≥ cả hai láng giềng" if comparison["keep"] else "< ít nhất một láng giềng"
        st.success(f"{outcome}: M(p) {relation} theo phép so sánh thực. NMS(p)={arrays['nms'][point]:.3f}.")


def _bfs_controls(arrays: dict) -> dict:
    replay = arrays["replay"]
    state = st.session_state
    state.canny_bfs_step = min(int(state.canny_bfs_step), len(replay.events))
    checkpoints = (0,) + tuple(i for i, (_, added) in enumerate(replay.events, 1) if added)
    if checkpoints[-1] != len(replay.events):
        checkpoints += (len(replay.events),)
    actions = (("Start / Reset", 0), ("← BFS", -1), ("Next BFS Step →", 1),
               ("Run 20 steps", 20), ("Run to completion", -2))
    with st.container(key="canny_bfs_controls"):
        cols = st.columns(5)
        for col, (label, delta) in zip(cols, actions):
            col.button(label, key=f"canny_bfs_{delta}", width="stretch",
                       disabled=delta == -1 and state.canny_bfs_step == 0,
                       on_click=_set_bfs, args=(delta, checkpoints))
    frame = replay_frame(replay, int(state.canny_bfs_step))
    st.caption(f"BFS event {state.canny_bfs_step}/{len(replay.events)} · "
               f"strong={int(arrays['strong'].sum())} · accepted weak={frame['accepted_weak']} · "
               f"frontier={int(frame['frontier'].sum())} · tổng accepted={int(frame['accepted'].sum())}.")
    st.caption("Next/Previous nhảy giữa các event phát hiện weak thật (bỏ qua queue pop không đổi ảnh). "
               "Event index là thứ tự BFS gốc; Run to completion vẫn xét hết queue.")
    if frame["active"] is not None:
        st.caption(f"Đang xử lý {frame['active']}; weak mới được nhận: {len(frame['added'])}. "
                   "Vàng vẫn là strong; trắng là weak đã được nhận.")
    return frame


def _set_bfs(delta: int, checkpoints: tuple[int, ...]) -> None:
    current = int(st.session_state.canny_bfs_step)
    if delta == 0:
        target = 0
    elif delta == -2:
        target = checkpoints[-1]
    elif delta > 0:
        following = [step for step in checkpoints if step > current]
        target = following[min(delta - 1, len(following) - 1)] if following else current
    else:
        preceding = [step for step in checkpoints if step < current]
        target = preceding[-1] if preceding else 0
    st.session_state.canny_bfs_step = target


def _stage_panel(arrays: dict, stage: str, *, point=None, comparison=None, frame=None,
                 title: str = "") -> None:
    presentation = bool(st.session_state.get("canny_presentation", False))
    with st.container(key="canny_panels"):
        left, right = st.columns([1.65, 1], gap="medium")
        with left:
            st.subheader(title or stage)
            st.plotly_chart(image_figure(arrays, stage, point=point, comparison=comparison, frame=frame,
                                         presentation=presentation), config=PLOT_CONFIG, key="canny_main_chart")
            st.caption("Cùng ảnh gốc, cùng tọa độ row/column ở mọi stage. Hover để đọc giá trị thật.")
        with right:
            if point is None:
                point = st.session_state.canny_point
            st.markdown(f"**Pixel đang theo dõi:** row {point[0]}, col {point[1]}")
            st.plotly_chart(image_figure(arrays, stage, point=point, comparison=comparison, frame=frame,
                                         presentation=presentation, patch=True), config=PLOT_CONFIG,
                            key="canny_stage_patch")
            st.caption("Patch 7×7 · ô vuông đúng pixel · không nội suy mờ.")


def guided_tutorial() -> None:
    _init()
    _header("Canny Edge Detector · Guided Tutorial", "Theo dõi một ảnh đi qua smoothing, gradient, NMS, threshold và hysteresis.")
    step = int(st.session_state.canny_guided_step)
    _progress(step)
    _navigation(step)
    _controls()
    arrays = _load()
    if arrays is None:
        return
    st.subheader(f"Bước {step}/14 · {STEPS[step - 1]}")
    point = st.session_state.canny_point
    if step in (1, 2):
        _stage_panel(arrays, "Original", point=point, title="Ảnh gốc · cấu trúc và nhiễu")
        st.info("Một pixel riêng lẻ không cho biết đâu là biên vật thể; texture và noise cũng gây thay đổi intensity."
                if step == 1 else "Cùng một ảnh: Original → Smooth → Gradient → NMS → Threshold → Hysteresis → Final.")
        _point_controls(arrays, key="canny_guided_point")
    elif step == 3:
        _stage_panel(arrays, "Original", point=point)
        st.plotly_chart(kernel_figure(arrays["kernel"]), config=PLOT_CONFIG)
        st.caption(f"Gaussian 5×5 · σ={st.session_state.canny_sigma:.2f} · tổng trọng số={arrays['kernel'].sum():.6f} · biên reflect.")
    elif step == 4:
        _stage_panel(arrays, "Smoothed", point=point, title="Gaussian smoothing")
        st.caption(f"Iσ = Gσ * I. Ảnh gốc tại p={arrays['original'][point]:.4f}; sau smoothing={arrays['smoothed'][point]:.4f}. Tăng σ giảm noise nhưng cũng có thể mất chi tiết mảnh.")
    elif step in (5, 6, 7):
        stage = "Gx" if step == 5 else "Magnitude"
        _stage_panel(arrays, stage, point=point)
        gx, gy = arrays["gx"][point], arrays["gy"][point]
        raw, normalized = arrays["magnitude_raw"][point], arrays["magnitude"][point]
        theta = "Undefined" if raw <= 1e-12 else f"{np.rad2deg(arrays['direction'][point]):.2f}°"
        st.markdown(f"**Gx={gx:.4f}, Gy={gy:.4f}** · M_raw={raw:.4f} · M_norm={normalized:.4f} · θ={theta}")
        st.caption(f"M_norm = clip(M_raw / P99.5, 0, 1); P99.5={arrays['magnitude_scale']:.4f}. Gx/Gy là signed response; θ theo hàng ảnh tăng xuống. Vector trên trục y-up là (Gx, −Gy).")
        if step == 5:
            st.plotly_chart(image_figure(arrays, "Gy", point=point, patch=True), config=PLOT_CONFIG)
        if step == 7:
            st.info("Nhiều pixel kề nhau có M lớn trên cùng một sườn biên. Ta cần so sánh dọc hướng gradient để làm mảnh.")
        _point_controls(arrays, key="canny_gradient_point")
    elif step in (8, 9):
        name = "keep" if step == 8 else "suppress"
        example = nms_examples(arrays)[name]
        selection_key = (step, st.session_state.canny_signature)
        if example is not None and st.session_state.get("canny_guided_example_key") != selection_key:
            st.session_state.canny_point = example
            st.session_state.canny_guided_example_key = selection_key
        point = st.session_state.canny_point
        comparison = inspect_nms(arrays["magnitude"], arrays["direction"], point)
        _stage_panel(arrays, "Magnitude", point=point, comparison=comparison,
                     title="NMS · p, q1, q2 trên magnitude trước NMS")
        st.caption(f"p={point}, q1={comparison['q1']}, q2={comparison['q2']} · θ={comparison['raw_degrees']:.1f}° → {comparison['degrees']}° · "
                   f"M(q1)={comparison['m1']:.3f}, M(p)={comparison['magnitude']:.3f}, M(q2)={comparison['m2']:.3f}")
        _prediction(step, f"{'KEEP' if comparison['keep'] else 'SUPPRESS'} vì M(p) "
                     f"{'≥ cả hai láng giềng' if comparison['keep'] else 'nhỏ hơn ít nhất một láng giềng'}; "
                     f"NMS(p)={arrays['nms'][point]:.3f}.")
        with st.expander("NMS microscope · chọn pixel khác"):
            _nms_inspector(arrays, gate=True)
    elif step == 10:
        _stage_panel(arrays, "NMS", point=point, title="NMS đã chuẩn hóa lần hai")
        st.caption("Quyết định KEEP/SUPPRESS dùng M_norm trước NMS. Ảnh hiển thị ở đây là NMS / max(NMS), dùng cho threshold sau đó; hai thang không tráo đổi trực tiếp.")
        st.info("NMS dùng 4 sector 0°, 45°, 90°, 135° và dấu ≥. Plateau bằng nhau có thể còn nhiều pixel; không bảo đảm mọi biên rộng đúng một pixel.")
    elif step == 11:
        _stage_panel(arrays, "Double Threshold", point=point)
        st.markdown(f"**Strong** {arrays['strong'].sum()} · **Weak** {arrays['weak'].sum()} · **Rejected/background** {arrays['weak'].size-arrays['strong'].sum()-arrays['weak'].sum()}")
        st.caption("Strong: NMS_norm ≥ high. Weak: low ≤ NMS_norm < high. Zero background không thuộc lớp nào.")
        _prediction(11, "Không. Weak chỉ được giữ khi có đường 8-connected đi tới strong; weak cô lập sẽ bị loại.")
    elif step == 12:
        frame = _bfs_controls(arrays)
        _stage_panel(arrays, "Hysteresis Replay", point=frame["active"] or point, frame=frame,
                     title="BFS từ toàn bộ strong pixels")
        st.info("Mỗi Next xử lý một phần tử queue thật. Màu trắng vẫn là weak (không biến thành strong). Run to completion khớp edge map batch.")
    elif step == 13:
        examples = weak_examples(arrays)
        isolated = examples["isolated"]
        connected = examples["connected"]
        point = isolated or connected or point
        _stage_panel(arrays, "Double Threshold", point=point,
                     title="Weak tách rời không được hỗ trợ bởi strong")
        st.markdown(f"Connected weak thực: **{connected if connected is not None else 'không có'}** → "
                    f"{'KEEP' if connected is not None else 'không áp dụng'}  \n"
                    f"Isolated weak thực: **{isolated if isolated is not None else 'không có'}** → "
                    f"{'REJECT' if isolated is not None else 'không áp dụng'}")
        st.caption("Ví dụ được tìm lại từ masks của ảnh và ngưỡng hiện tại; nếu không tồn tại, app không dựng nhãn giả.")
    else:
        _stage_panel(arrays, "Final Edge Map", point=point, title="Biên nhị phân cuối cùng")
        st.caption(f"Strong={arrays['strong'].sum()} · weak được nối={np.sum(arrays['weak'] & arrays['edges'])} · weak cô lập={np.sum(arrays['weak'] & ~arrays['edges'])}.")
        _prediction(14, "Không. Noise, blur, ngưỡng sai, tương phản yếu và texture vẫn có thể tạo hoặc làm mất biên. Canny tìm cấu trúc tín hiệu, không hiểu vật thể.")


def experiment_lab() -> None:
    _init()
    _header("Canny Edge Detector · Experiment Lab", "Đổi ảnh và tham số, kiểm tra NMS hoặc tự chạy từng bước hysteresis trên cùng pipeline.")
    _controls(lab=True)
    arrays = _load()
    if arrays is None:
        return
    st.caption(f"{st.session_state.canny_preset} · σ={st.session_state.canny_sigma:.2f} · "
               f"low={st.session_state.canny_low:.2f} · high={st.session_state.canny_high:.2f}")
    with st.expander("Dự đoán cho preset · Observe → Predict → Explain"):
        st.markdown(f'<div class="question"><b>Câu hỏi preset:</b> {escape(PRESET_QUESTIONS[st.session_state.canny_preset])}</div>', unsafe_allow_html=True)
        st.radio("Dự đoán trước khi xem kết quả", ("Biên sẽ rõ hơn", "Biên sẽ yếu/mất", "Phụ thuộc kết nối và ngưỡng"),
                 index=None, key="canny_preset_predict")
        st.button("Reveal preset takeaway", disabled=not st.session_state.get("canny_preset_predict"),
                  on_click=lambda: st.session_state.__setitem__("canny_preset_reveal", True))
        if st.session_state.get("canny_preset_reveal"):
            st.success(f"Từ phép tính hiện tại: strong={arrays['strong'].sum()}, connected weak={(arrays['weak'] & arrays['edges']).sum()}, "
                       f"isolated weak={(arrays['weak'] & ~arrays['edges']).sum()}. "
                       f"{PRESET_TAKEAWAYS[st.session_state.canny_preset]} "
                       "Đổi σ/ngưỡng để kiểm tra giới hạn, không có đáp án cố định cho mọi cấu hình.")
    st.radio("Workspace", ("A · Pipeline Explorer", "B · NMS & Hysteresis Explorer"),
             key="canny_workspace", horizontal=True)
    if st.session_state.canny_workspace.startswith("A"):
        stage = st.selectbox("Pipeline stage", STAGES, key="canny_lab_stage")
        point = _point_controls(arrays, key="canny_lab_point")
        _stage_panel(arrays, stage, point=point)
        if stage != "Original":
            with st.expander("So sánh ảnh gốc"):
                st.plotly_chart(image_figure(arrays, "Original", point=point), config=PLOT_CONFIG)
        value = {
            "Original": arrays["original"], "Smoothed": arrays["smoothed"],
            "Gx": arrays["gx"], "Gy": arrays["gy"], "Magnitude": arrays["magnitude"],
            "Gradient Direction": np.rad2deg(arrays["direction"]), "NMS": arrays["nms_normalized"],
            "Double Threshold": arrays["strong"].astype(int) * 2 + arrays["weak"].astype(int),
            "Final Edge Map": arrays["edges"].astype(int),
        }[stage][point]
        st.caption(f"Stage={stage} · p={point} · giá trị thật={value:.5f}. Gx/Gy signed raw; Magnitude và NMS là hai chuẩn hóa riêng.")
        st.markdown(f"**Counts:** strong {arrays['strong'].sum()} · weak {arrays['weak'].sum()} · "
                    f"accepted weak {(arrays['weak'] & arrays['edges']).sum()} · final {arrays['edges'].sum()} pixels.")
    else:
        mode = st.radio("Explorer", ("NMS Microscope", "Hysteresis BFS"), key="canny_explorer", horizontal=True)
        if mode == "NMS Microscope":
            point = st.session_state.canny_point
            comparison = inspect_nms(arrays["magnitude"], arrays["direction"], point)
            _stage_panel(arrays, "Magnitude", point=point, comparison=comparison,
                         title="NMS · magnitude trước suppression")
            _nms_inspector(arrays)
        else:
            frame = _bfs_controls(arrays)
            _stage_panel(arrays, "Hysteresis Replay", point=frame["active"] or st.session_state.canny_point,
                         frame=frame, title="BFS thật · toàn ảnh")
            examples = weak_examples(arrays)
            st.caption(f"Connected weak={examples['connected']} · isolated weak={examples['isolated']}. "
                       "Frontier trong hình là queue chưa pop; replay không tự chạy khi Streamlit rerun.")
