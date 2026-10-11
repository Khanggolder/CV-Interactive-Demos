"""Hough line-transform Guided Tutorial and Experiment Lab."""

from __future__ import annotations

import streamlit as st

from utils.hough import (
    ALIGNED, COMPETING, DEFAULT_POINTS, DISTRACTORS, accumulate, empty_votes,
    line_from_points, make_grid, peak_geometry, peaks, signed_rho,
    vote_all, vote_next, vote_remaining, VoteState,
)
from utils.hough_visualization import accumulator_figure, continuous_figure, image_figure
from utils.visualization import PLOT_CONFIG


STEPS = (
    "Edge pixels → lines", "Một điểm, nhiều đường", "Normal Form",
    "Điểm → sinusoid", "Hai đường cong", "Giao điểm → một line",
    "Sáu điểm thẳng hàng", "Rời rạc hóa", "Một điểm vote",
    "Nhiều điểm tạo peak", "Peak → detected line", "Giới hạn và kết luận",
)
PRESETS = {
    "A · Sáu điểm thẳng hàng": (ALIGNED, 10.0, 0.5,
        "Sáu điểm cùng line sẽ dồn vote vào đâu?"),
    "B · Sáu điểm + nhiễu": (DEFAULT_POINTS, 10.0, 0.5,
        "Ba distractor có xóa mất peak chính không?"),
    "C · Hai line cạnh tranh": (COMPETING, 10.0, 0.5,
        "Nếu hai line có support gần nhau, peak cao nhất có kể hết câu chuyện không?"),
    "D · Lưới thô": (DEFAULT_POINTS, 30.0, 1.0,
        "Cùng điểm, bin θ/ρ rộng hơn sẽ thay đổi peak và sai lệch ra sao?"),
}


def _init_guided() -> None:
    st.session_state.setdefault("hough_guided_step", 1)
    st.session_state.setdefault("hough_guided_theta", 0.0)
    st.session_state.setdefault("hough_guided_reveal", False)
    st.session_state.setdefault("hough_guided_votes", empty_votes(DEFAULT_POINTS, make_grid()))


def _guided_move(offset: int) -> None:
    st.session_state.hough_guided_step = min(12, max(1, st.session_state.hough_guided_step + offset))


def _guided_reset() -> None:
    st.session_state.hough_guided_step = 1
    st.session_state.hough_guided_theta = 0.0
    st.session_state.hough_guided_reveal = False
    st.session_state.hough_guided_votes = empty_votes(DEFAULT_POINTS, make_grid())
    st.session_state.pop("hough_guided_prediction", None)


def _guide_vote(action: str) -> None:
    grid = make_grid()
    state = st.session_state.hough_guided_votes
    if action == "next":
        state = vote_next(state, DEFAULT_POINTS, grid, 0)
    elif action == "selected":
        state = vote_remaining(state, DEFAULT_POINTS, grid, 0)
    elif action == "aligned":
        for index in range(len(ALIGNED)):
            state = vote_remaining(state, DEFAULT_POINTS, grid, index)
    elif action == "all":
        state = vote_all(state, DEFAULT_POINTS, grid)
    st.session_state.hough_guided_votes = state


def _header(title: str, purpose: str) -> None:
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("hough_presentation", False):
        st.markdown(f'<div class="hero-copy">{purpose}</div>', unsafe_allow_html=True)


def _panels(points, *, active=0, theta=None, curves=1, accumulator=None, grid=None,
            show_peak=False, detected=None, ideal=False, visible=None, intersection=False,
            image_only=False) -> None:
    presentation = bool(st.session_state.get("hough_presentation", False))
    if image_only:
        st.plotly_chart(image_figure(points, active=active, theta=theta, visible=visible,
                                     detected=detected, ideal=ideal, presentation=presentation),
                        config=PLOT_CONFIG)
        return
    with st.container(key="hough_panels"):
        left, right = st.columns(2, gap="medium")
        with left:
            st.plotly_chart(image_figure(points, active=active, theta=theta, visible=visible,
                                         detected=detected, ideal=ideal, presentation=presentation),
                            config=PLOT_CONFIG)
        with right:
            if accumulator is None:
                figure = continuous_figure(points, active=active, theta=theta or 0.0,
                                           curves=curves, intersection=intersection,
                                           presentation=presentation)
            else:
                figure = accumulator_figure(accumulator.counts, grid,
                                             active_cell=accumulator.last_vote[1:] if accumulator.last_vote and not show_peak else None,
                                             show_peak=show_peak, presentation=presentation)
            st.plotly_chart(figure, config=PLOT_CONFIG)


def _guided_progress(step: int) -> None:
    cells = ''.join(
        f'<div class="step {"active" if index == step else "done" if index < step else ""}">{index}<br>{label}</div>'
        for index, label in enumerate(("Pixels", "Lines", "Normal", "Curve", "Two", "Meet",
                                       "Aligned", "Bins", "Vote", "Peak", "Detect", "Limits"), 1)
    )
    st.markdown(f'<div class="stepper hough-stepper">{cells}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-progress">Tiến độ học · {step}/12'
                f'<div class="progress-track"><span style="width:{step / 12 * 100:.1f}%"></span></div></div>',
                unsafe_allow_html=True)


def _theta_control(key: str) -> float:
    return float(st.slider("θ · góc của normal (°)", 0.0, 179.9, step=0.1, key=key))


def _peak_text(counts, grid) -> tuple[tuple[float, float] | None, str]:
    maximum, winners = peaks(counts)
    if not winners:
        return None, "Chưa có vote; chưa có peak."
    row, column = winners[0]
    rho, theta = peak_geometry(grid, row, column)
    tie = f"Có {len(winners)} ô đồng hạng; chưa có line duy nhất." if len(winners) > 1 else "Peak duy nhất."
    return (rho, theta) if len(winners) == 1 else None, (
        f"Peak A[{row},{column}] = {maximum} vote · "
        f"θ ∈ [{grid.theta_edges[column]:g}°, {grid.theta_edges[column+1]:g}°), "
        f"ρ ∈ [{grid.rho_edges[row]:g}, {grid.rho_edges[row+1]:g}) · "
        f"bin center (θ,ρ)=({theta:g}°, {rho:g}). {tie}"
    )


def guided_tutorial() -> None:
    _init_guided()
    _header("Hough Transform · Guided Tutorial", "Một edge point mở ra nhiều line; nhiều point cùng line gặp nhau ở Hough Space.")
    step = int(st.session_state.hough_guided_step)
    with st.container(key="hough_nav"):
        previous, following, reset = st.columns(3)
        previous.button("← Previous", disabled=step == 1, on_click=_guided_move, args=(-1,), width="stretch")
        gated = step == 5 and not st.session_state.hough_guided_reveal
        following.button("Next →", disabled=step == 12 or gated, on_click=_guided_move, args=(1,), width="stretch")
        reset.button("Reset", on_click=_guided_reset, width="stretch")
    _guided_progress(step)
    st.subheader(f"Bước {step}/12 · {STEPS[step-1]}")
    grid = make_grid()
    state = st.session_state.hough_guided_votes
    theta = float(st.session_state.hough_guided_theta)
    true_rho, true_theta = line_from_points(DEFAULT_POINTS[0], DEFAULT_POINTS[1])

    if step == 1:
        _panels(DEFAULT_POINTS, image_only=True)
        st.markdown('<div class="question">Hai điểm xác định một line. Nhưng nếu có nhiều edge point không liên quan thì chọn cặp nào?</div>', unsafe_allow_html=True)
        st.caption("Quan sát 6 điểm trên x + 2y = 5 và 3 distractor; màu chưa nói trước point nào thuộc line.")
    elif step in (2, 3, 4):
        theta = _theta_control("hough_guided_theta")
        if step == 2:
            _panels(DEFAULT_POINTS, theta=theta, visible=1, image_only=True)
            st.caption("Đổi θ: candidate luôn đi qua p1 nhưng đổi hướng. Một điểm cho vô số line có thể.")
        elif step == 3:
            _panels(DEFAULT_POINTS, theta=theta, visible=1, image_only=True)
            st.latex(r"\rho = x\cos\theta + y\sin\theta")
            st.caption(f"p1=(1,2), θ={theta:.1f}°, signed ρ={signed_rho(DEFAULT_POINTS[0], theta):.4f}. "
                       "Đường cam là normal có hướng; ρ âm vẫn hợp lệ khi hình chiếu ngược hướng normal.")
        else:
            _panels(DEFAULT_POINTS, theta=theta, visible=1)
            st.caption(f"Cùng θ={theta:.1f}° cho candidate bên trái và marker "
                       f"(θ,ρ)=({theta:.1f}°, {signed_rho(DEFAULT_POINTS[0], theta):.4f}) bên phải. "
                       "Quét θ để thấy p1 vẽ một sinusoid thật.")
    elif step == 5:
        theta = _theta_control("hough_guided_theta")
        st.radio("Dự đoán: giao điểm của hai sinusoid có nghĩa gì?",
                 ("Một line đi qua cả hai điểm", "Hai line không liên quan", "Chưa chắc"),
                 index=None, key="hough_guided_prediction", horizontal=True)
        st.button("Reveal giải thích", disabled=not st.session_state.get("hough_guided_prediction"),
                  on_click=lambda: st.session_state.update(hough_guided_reveal=True))
        _panels(DEFAULT_POINTS, theta=theta, curves=2, visible=2,
                intersection=bool(st.session_state.hough_guided_reveal))
        if st.session_state.hough_guided_reveal:
            st.success("Cùng (θ,ρ) nghĩa là hai điểm có cùng hình chiếu lên normal: một line đi qua cả hai.")
    elif step == 6:
        _panels(DEFAULT_POINTS, theta=true_theta, curves=2, visible=2,
                intersection=True, detected=(true_rho, true_theta))
        st.caption(f"Giải độc lập từ p1 và p2: θ*={true_theta:.6f}° là góc NORMAL, "
                   f"ρ*=√5={true_rho:.6f}. Line trong Image Space: x + 2y = 5.")
    elif step == 7:
        _panels(DEFAULT_POINTS, theta=true_theta, curves=6, visible=6,
                intersection=True, detected=(true_rho, true_theta))
        st.caption("Bốn point thẳng hàng mới tạo bốn sinusoid cũng đi qua (θ*,ρ*). "
                   "Giao điểm chung mạnh hơn vì có nhiều point ủng hộ.")
    elif step == 8:
        st.latex(r"\Delta\theta=10^\circ,\quad\Delta\rho=0.5,\quad A[\mathrm{rho\ row},\mathrm{theta\ column}]")
        _panels(DEFAULT_POINTS, theta=true_theta, visible=6,
                accumulator=empty_votes(DEFAULT_POINTS, grid), grid=grid)
        st.caption("Lưới 18 cột × 20 hàng; θ sample ở tâm bin 5°,15°,…,175°. "
                   "Mỗi cell là một khoảng nửa mở, không phải giao điểm liên tục chính xác.")
    elif step in (9, 10, 11):
        if step == 9:
            a, b = st.columns(2)
            a.button("Next Vote", on_click=_guide_vote, args=("next",),
                     disabled=state.next_indices[0] == grid.shape[1], width="stretch")
            b.button("Vote Selected Point", on_click=_guide_vote, args=("selected",),
                     disabled=state.next_indices[0] == grid.shape[1], width="stretch")
            st.caption(f"p1 đã cast {state.next_indices[0]}/{grid.shape[1]} vote. "
                       "Mỗi lần chỉ cộng một cell tại θ sample kế tiếp; rerun không vote thêm.")
        elif step == 10:
            a, b = st.columns(2)
            a.button("Vote 6 aligned", on_click=_guide_vote, args=("aligned",), width="stretch")
            b.button("Vote All 9", on_click=_guide_vote, args=("all",), width="stretch")
            st.caption("Thêm 6 point cùng line rồi 3 distractor. Nhiễu thường rải phiếu, "
                       "nhưng vẫn có thể tạo peak cạnh tranh.")
        else:
            if sum(state.next_indices) < len(DEFAULT_POINTS) * grid.shape[1]:
                st.button("Hoàn tất 9 điểm để phát hiện", on_click=_guide_vote, args=("all",))
            st.caption("Đường xanh lá lấy từ BIN CENTER của peak; đường xám đứt là source line lý tưởng.")
        state = st.session_state.hough_guided_votes
        detected, description = _peak_text(state.counts, grid)
        active = state.last_vote[0] if state.last_vote else 0
        current_theta = grid.theta_centers[state.last_vote[2]] if state.last_vote else None
        _panels(DEFAULT_POINTS, active=active, theta=current_theta if step != 11 else None,
                accumulator=state, grid=grid,
                show_peak=step == 11, detected=detected if step == 11 else None,
                ideal=step == 11)
        st.metric("Votes đã cast", f"{int(state.counts.sum())}/{len(DEFAULT_POINTS)*grid.shape[1]}")
        if state.last_vote:
            p, row, col = state.last_vote
            st.caption(f"Vote cuối: p{p+1}, θ={grid.theta_centers[col]:g}°, "
                       f"ρ={signed_rho(DEFAULT_POINTS[p], grid.theta_centers[col]):.4f} → "
                       f"A[{row},{col}]={state.counts[row,col]}.")
        if step == 11:
            st.info(description)
            if detected is not None:
                deviations = [abs(signed_rho(point, detected[1]) - detected[0]) for point in ALIGNED]
                st.caption(f"Bin-center line: x cos {detected[1]:g}° + y sin {detected[1]:g}° = {detected[0]:g}. "
                           f"Max khoảng cách của 6 point lý tưởng tới line này: {max(deviations):.3f}; "
                           "đó là quantization error, không phải TLS refit.")
    else:
        st.markdown('<div class="question">Độ phân giải, clutter, support và threshold có thể làm kết luận đổi ra sao?</div>', unsafe_allow_html=True)
        scenario = st.selectbox("Thử giới hạn", ("θ quá thô", "ρ quá thô", "Ít support", "Outlier clutter", "Threshold"),
                                key="hough_limit")
        choices = {"θ quá thô": (DEFAULT_POINTS, 45.0, 0.5),
                   "ρ quá thô": (DEFAULT_POINTS, 10.0, 2.0),
                   "Ít support": (DEFAULT_POINTS[:2] + DISTRACTORS, 10.0, 0.5),
                   "Outlier clutter": (COMPETING, 10.0, 0.5),
                   "Threshold": (DEFAULT_POINTS, 10.0, 0.5)}
        points, dtheta, drho = choices[scenario]
        threshold = st.slider("Peak threshold (votes)", 1, len(points) + 1, 4, key="hough_limit_threshold")
        test_grid = make_grid(dtheta, drho, points)
        counts = accumulate(points, test_grid)
        maximum, winners = peaks(counts)
        test_state = VoteState(counts, (test_grid.shape[1],) * len(points))
        detected, description = _peak_text(counts, test_grid)
        _panels(points, accumulator=test_state, grid=test_grid, show_peak=True,
                detected=detected if maximum >= threshold else None)
        st.info(description)
        st.caption(f"Max={maximum}, threshold={threshold}: " +
                   ("qua ngưỡng; vẫn cần xét peak cạnh tranh." if maximum >= threshold else "không phát hiện line ở ngưỡng này."))
        st.markdown('<div class="takeaway">Hough hữu ích khi nhiều edge point ủng hộ cấu trúc chung. '
                    'Bin thô giảm chi tiết; bin mịn phân tán vote; clutter có thể tạo peak giả; '
                    'threshold quá thấp dễ nhận nhiễu, quá cao bỏ lỡ line yếu.</div>', unsafe_allow_html=True)


def _init_lab() -> None:
    st.session_state.setdefault("hough_lab_points", list(DEFAULT_POINTS))
    st.session_state.setdefault("hough_lab_custom", [])
    st.session_state.setdefault("hough_lab_base_count", len(DEFAULT_POINTS))
    st.session_state.setdefault("hough_lab_point", 0)
    st.session_state.setdefault("hough_lab_theta", 0.0)
    st.session_state.setdefault("hough_lab_theta_width", 10.0)
    st.session_state.setdefault("hough_lab_rho_width", 0.5)
    st.session_state.setdefault("hough_lab_workspace", "A · Point → Sinusoid")
    st.session_state.setdefault("hough_lab_loaded", "B · Sáu điểm + nhiễu")
    st.session_state.setdefault("hough_lab_preset", "B · Sáu điểm + nhiễu")


def _lab_reset_votes() -> None:
    points = st.session_state.hough_lab_points
    grid = make_grid(st.session_state.hough_lab_theta_width, st.session_state.hough_lab_rho_width, points)
    st.session_state.hough_lab_votes = empty_votes(points, grid)
    st.session_state.hough_lab_signature = (tuple(map(tuple, points)),
                                            st.session_state.hough_lab_theta_width,
                                            st.session_state.hough_lab_rho_width)


def _load_preset() -> None:
    name = st.session_state.hough_lab_preset
    points, theta_width, rho_width, _ = PRESETS[name]
    st.session_state.hough_lab_points = list(points)
    st.session_state.hough_lab_custom = []
    st.session_state.hough_lab_base_count = len(points)
    st.session_state.hough_lab_point = 0
    st.session_state.hough_lab_theta_width = theta_width
    st.session_state.hough_lab_rho_width = rho_width
    st.session_state.hough_lab_loaded = name
    st.session_state.hough_lab_prediction = None
    _lab_reset_votes()


def _default_lab() -> None:
    st.session_state.hough_lab_preset = "B · Sáu điểm + nhiễu"
    _load_preset()


def _add_point() -> None:
    point = (float(st.session_state.hough_add_x), float(st.session_state.hough_add_y))
    st.session_state.hough_lab_points.append(point)
    st.session_state.hough_lab_custom.append(point)
    st.session_state.hough_lab_point = len(st.session_state.hough_lab_points) - 1
    _lab_reset_votes()


def _remove_point() -> None:
    index = int(st.session_state.hough_lab_point)
    if index >= st.session_state.hough_lab_base_count:
        st.session_state.hough_lab_custom.pop(index - st.session_state.hough_lab_base_count)
        st.session_state.hough_lab_points.pop(index)
        st.session_state.hough_lab_point = min(index, len(st.session_state.hough_lab_points) - 1)
        _lab_reset_votes()


def _lab_vote(action: str) -> None:
    points = st.session_state.hough_lab_points
    grid = make_grid(st.session_state.hough_lab_theta_width, st.session_state.hough_lab_rho_width, points)
    state = st.session_state.hough_lab_votes
    index = int(st.session_state.hough_lab_point)
    if action == "next":
        state = vote_next(state, points, grid, index)
    elif action == "selected":
        state = vote_remaining(state, points, grid, index)
    elif action == "all":
        state = vote_all(state, points, grid)
    st.session_state.hough_lab_votes = state


def experiment_lab() -> None:
    _init_lab()
    _header("Hough Transform · Experiment Lab", "Dịch θ để nhìn point → sinusoid; cast từng vote để tự xây Accumulator.")
    st.radio("Workspace", ("A · Point → Sinusoid", "B · Accumulator Voting"),
             key="hough_lab_workspace", horizontal=True)
    visual_slot = st.container()
    st.selectbox("Teaching preset", tuple(PRESETS), key="hough_lab_preset")
    preset_col, default_col = st.columns(2)
    preset_col.button("Nạp preset / Reset points", on_click=_load_preset, width="stretch")
    default_col.button("Về ví dụ mặc định", on_click=_default_lab, width="stretch")
    loaded = st.session_state.hough_lab_loaded
    st.markdown(f'<div class="question"><b>Dự đoán:</b> {PRESETS[loaded][3]}</div>', unsafe_allow_html=True)
    st.radio("Peak sẽ như thế nào?", ("Một peak rõ", "Nhiều peak cạnh tranh", "Khó phát hiện"),
             index=None, key="hough_lab_prediction", horizontal=True)
    points = st.session_state.hough_lab_points
    point_labels = [f"p{i+1} · ({p[0]:g}, {p[1]:g})" for i, p in enumerate(points)]
    st.selectbox("Active point", range(len(points)), format_func=lambda index: point_labels[index],
                 key="hough_lab_point")
    theta = _theta_control("hough_lab_theta")
    with st.expander("Thêm / bỏ custom point", expanded=False):
        left, right = st.columns(2)
        left.number_input("x", -20.0, 20.0, 0.0, step=0.1, key="hough_add_x")
        right.number_input("y", -20.0, 20.0, 0.0, step=0.1, key="hough_add_y")
        a, b = st.columns(2)
        a.button("Thêm point", on_click=_add_point, width="stretch")
        b.button("Bỏ custom point đang chọn", on_click=_remove_point,
                 disabled=st.session_state.hough_lab_point < st.session_state.hough_lab_base_count,
                 width="stretch")
        st.caption("Đổi tập point sẽ reset Accumulator; điểm gốc của preset không bị xóa.")
    with st.expander("Bin resolution · Δθ / Δρ", expanded=False):
        a, b = st.columns(2)
        a.select_slider("Δθ (°)", (2.0, 5.0, 10.0, 15.0, 30.0, 45.0), key="hough_lab_theta_width")
        b.select_slider("Δρ", (0.25, 0.5, 1.0, 2.0), key="hough_lab_rho_width")
        st.caption("Thay bin width sẽ reset votes. θ luôn chia hết [0°,180°); ρ có thể tự mở rộng đối xứng khi thêm point xa.")
    points = st.session_state.hough_lab_points
    signature = (tuple(map(tuple, points)), st.session_state.hough_lab_theta_width,
                 st.session_state.hough_lab_rho_width)
    invalidated = signature != st.session_state.get("hough_lab_signature")
    if invalidated:
        _lab_reset_votes()
        st.info("Tập point hoặc bin geometry đã đổi: Accumulator cũ được reset để tránh count sai cấu hình.")
    grid = make_grid(st.session_state.hough_lab_theta_width, st.session_state.hough_lab_rho_width, points)
    if grid.rho_edges[-1] > 5.0:
        st.caption(f"ρ range tự mở rộng thành [{grid.rho_edges[0]:g}, {grid.rho_edges[-1]:g}) "
                   "để mọi projection nằm trong lưới; không cắt vote vào biên.")
    active = int(st.session_state.hough_lab_point)
    rho = signed_rho(points[active], theta)
    state = st.session_state.hough_lab_votes
    if st.session_state.hough_lab_workspace.startswith("A"):
        with visual_slot:
            _panels(points, active=active, theta=theta, curves=len(points))
        a, b = st.columns(2)
        a.metric("θ · normal", f"{theta:.1f}°")
        b.metric("Signed ρ", f"{rho:.4f}")
        st.caption("Điểm vàng, candidate line và marker vàng cùng dùng một θ và một phép chiếu ρ. "
                   "Chọn point khác hoặc thêm point để xem sinusoid tương ứng.")
    else:
        threshold = st.slider("Peak threshold · vote tối thiểu", 1, max(4, len(points) + 1),
                              value=4, key="hough_lab_threshold")
        done = state.next_indices[active] >= grid.shape[1]
        with st.container(key="hough_vote_actions"):
            a, b, c, d = st.columns(4)
            a.button("Next Vote", disabled=done, on_click=_lab_vote, args=("next",), width="stretch")
            b.button("Vote Selected", disabled=done, on_click=_lab_vote, args=("selected",), width="stretch")
            c.button("Vote All", on_click=_lab_vote, args=("all",), width="stretch")
            d.button("Reset Accumulator", on_click=_lab_reset_votes, width="stretch")
        state = st.session_state.hough_lab_votes
        maximum, winners = peaks(state.counts)
        detected, description = _peak_text(state.counts, grid)
        accepted = maximum >= threshold
        with visual_slot:
            _panels(points, active=active, theta=theta, accumulator=state, grid=grid,
                    show_peak=accepted, detected=detected if accepted else None,
                    ideal=loaded != "C · Hai line cạnh tranh" and not st.session_state.hough_lab_custom)
        metrics = st.columns(3)
        metrics[0].metric("Total votes", f"{int(state.counts.sum())}/{len(points)*grid.shape[1]}")
        metrics[1].metric("Selected point", f"{state.next_indices[active]}/{grid.shape[1]}")
        metrics[2].metric("Current peak", f"{maximum} vote")
        if state.last_vote:
            index, row, col = state.last_vote
            st.caption(f"Vote cuối: p{index+1}, θ={grid.theta_centers[col]:g}°, "
                       f"ρ={signed_rho(points[index], grid.theta_centers[col]):.4f} → "
                       f"A[{row},{col}]={state.counts[row,col]}.")
        st.info(description)
        if len(winners) > 1:
            st.caption("Các cell đồng hạng (rho row, theta column): "
                       + ", ".join(map(str, winners[:8]))
                       + (" …" if len(winners) > 8 else ""))
        if not accepted:
            st.warning(f"Peak {maximum} < threshold {threshold}: chưa công bố detected line.")
        elif len(winners) > 1:
            st.warning("Nhiều cell cùng đạt max; không tự ý chọn một line duy nhất.")
        elif int(state.counts.sum()) < len(points)*grid.shape[1]:
            st.caption("Đây là peak TẠM THỜI vì chưa cast đủ vote; không phải kết quả cuối.")
        if st.session_state.get("hough_lab_prediction"):
            st.markdown('<div class="takeaway"><b>Quan sát:</b> Count = số point đã vote vào từng cell. '
                        'Line phát hiện lấy từ bin center, nên có quantization error. '
                        'Peak cao nhất không tự chứng minh line đúng khi có clutter hoặc nhiều cấu trúc.</div>',
                        unsafe_allow_html=True)
            if int(state.counts.sum()) == len(points) * grid.shape[1]:
                non_winning = state.counts.copy()
                non_winning[tuple(zip(*winners))] = 0
                runner_up = int(non_winning.max(initial=0))
                st.caption(f"Kết quả {loaded}: max={maximum}, runner-up={runner_up}, "
                           f"{len(winners)} peak đồng hạng; threshold={threshold}. "
                           "Mỗi point góp một vote cho mỗi θ sample, nhưng nhiều line hoặc bin thô "
                           "có thể tạo peak cạnh tranh. Threshold thấp dễ nhận clutter; cao có thể bỏ line yếu.")
        if loaded.startswith("D") and int(state.counts.sum()) == len(points)*grid.shape[1]:
            fine = accumulate(points, make_grid(5.0, 0.25, points))
            fine_max, fine_winners = peaks(fine)
            st.caption(f"Cùng dữ liệu, lưới fine 5°/0.25: max={fine_max}, {len(fine_winners)} peak đồng hạng. "
                       f"Lưới hiện tại {grid.theta_edges[1]-grid.theta_edges[0]:g}°/"
                       f"{grid.rho_edges[1]-grid.rho_edges[0]:g}: max={maximum}. "
                       "Không so vote count như độ chính xác hình học tuyệt đối.")
