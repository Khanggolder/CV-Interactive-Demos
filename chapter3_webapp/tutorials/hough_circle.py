"""Guided Hough Circle geometry, voting replay and connected Experiment Lab."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import streamlit as st

from utils.hough_circle import (
    CENTER, RADIUS, PRESETS, Dataset, Observation, bin_index, candidate_center,
    cell_parameters, common_centers, empty_votes, generate_circle_example,
    gradient_candidates, make_grid, peaks, preset_dataset, top_cells, vote_all,
    vote_events, vote_next, vote_previous, vote_remaining, vote_signature,
)
from utils.hough_circle_visualization import center_figure, image_figure
from utils.visualization import PLOT_CONFIG

STEPS = (
    "Từ edge đến circle", "Một điểm, biết r", "Locus của tâm", "Hai điểm: còn mơ hồ",
    "Điểm thứ ba", "Sáu điểm ủng hộ", "Nếu chưa biết r?", "Không gian (a,b,r)",
    "Bỏ phiếu vào bin", "Nhiều điểm tạo peak", "Gradient thu hẹp tìm kiếm",
    "Polarity có thể đảo", "Khôi phục circle", "Thất bại và kết luận",
)


def _init() -> None:
    defaults = {
        "circle_guided_step": 1, "circle_guided_phi": -90.0,
        "circle_guided_radius": 2.0, "circle_guided_point": 0,
        "circle_guided_visible": 1, "circle_guided_reveal_4": False,
        "circle_guided_reveal_11": False, "circle_guided_votes": None,
        "circle_lab_preset": PRESETS[0], "circle_loaded_preset": PRESETS[0],
        "circle_lab_dataset": generate_circle_example(), "circle_lab_workspace": "A · Center Locus",
        "circle_lab_point": 0, "circle_lab_phi": -90.0, "circle_lab_radius": 2.0,
        "circle_lab_visible": 1, "circle_lab_show_locus": True,
        "circle_lab_search": "Known radius",
        "circle_lab_voting": "Full locus", "circle_lab_polarity": "Inward",
        "circle_lab_layer": 0, "circle_lab_votes": None, "circle_lab_signature": None,
        "circle_lab_detected": 0, "circle_lab_prediction": None,
        "circle_custom_serial": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _header(title, purpose):
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("circle_presentation", False):
        st.markdown(f'<div class="hero-copy">{purpose}</div>', unsafe_allow_html=True)


def _progress(step):
    cells = "".join(f'<div class="step {"active" if i == step else "done" if i < step else ""}">{i}<br>{label}</div>'
                    for i, label in enumerate(("Edge", "One", "Locus", "Two", "Three", "Six",
                                               "Radius", "3D", "Bin", "Peak", "Grad", "±", "Detect", "Limits"), 1))
    st.markdown(f'<div class="stepper circle-stepper">{cells}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-progress">Tiến độ học · {step}/14'
                f'<div class="progress-track"><span style="width:{step/14*100:.1f}%"></span></div></div>',
                unsafe_allow_html=True)


def _move_step(delta):
    st.session_state.circle_guided_step = min(14, max(1, st.session_state.circle_guided_step + delta))


def _reset_guided():
    st.session_state.circle_guided_step = 1
    st.session_state.circle_guided_phi = -90.0
    st.session_state.circle_guided_radius = 2.0
    st.session_state.circle_guided_point = 0
    st.session_state.circle_guided_visible = 1
    st.session_state.circle_guided_votes = None
    st.session_state.pop("circle_guided_phi_control", None)
    st.session_state.pop("circle_guided_radius_control", None)
    for key in ("circle_guided_predict_4", "circle_guided_predict_11"):
        st.session_state.pop(key, None)
    st.session_state.circle_guided_reveal_4 = False
    st.session_state.circle_guided_reveal_11 = False


def _nav(step):
    gated = (step == 4 and not st.session_state.circle_guided_reveal_4) or (
        step == 11 and not st.session_state.circle_guided_reveal_11)
    with st.container(key="circle_nav"):
        a, b, c = st.columns(3)
        a.button("← Previous", key="circle_previous", disabled=step == 1,
                 on_click=_move_step, args=(-1,), width="stretch")
        b.button("Next →", key="circle_next", disabled=step == 14 or gated,
                 on_click=_move_step, args=(1,), width="stretch")
        c.button("Reset", key="circle_reset", on_click=_reset_guided, width="stretch")


def _point_picker(dataset, key):
    st.selectbox("Edge point", range(len(dataset.points)),
                 format_func=lambda i: f"{dataset.points[i].id} · ({dataset.points[i].xy[0]:.3f}, {dataset.points[i].xy[1]:.3f})",
                 key=key, on_change=_show_active)


def _save_slider(model_key):
    st.session_state[model_key] = float(st.session_state[f"{model_key}_control"])


def _float_slider(label, minimum, maximum, step, model_key, *, disabled=False):
    st.slider(label, minimum, maximum, value=float(st.session_state[model_key]), step=step,
              key=f"{model_key}_control", disabled=disabled,
              on_change=_save_slider, args=(model_key,))


def _show_active():
    st.session_state.circle_lab_visible = max(int(st.session_state.circle_lab_visible),
                                               int(st.session_state.circle_lab_point)+1)


def _panels(dataset, *, active=0, visible=None, radius=2.0, phi=-90.0,
            locus=True, intersections=(), grid=None, votes=None, layer=0,
            show_peak=False, detected=None, truth=False, gradient=False,
            candidate=True, image_only=False):
    presentation = bool(st.session_state.get("circle_presentation", False))
    with st.container(key="circle_panels"):
        if image_only:
            with st.container(key="circle_image_only"):
                st.plotly_chart(image_figure(dataset, active=active, visible=visible, radius=radius,
                                             phi=phi, candidate=False, show_truth=truth,
                                             presentation=presentation),
                                config=PLOT_CONFIG, key="circle_image_chart")
            return
        left, right = st.columns(2, gap="medium")
        with left:
            st.plotly_chart(image_figure(dataset, active=active, visible=visible, radius=radius,
                                         phi=phi, candidate=candidate, detected=detected, show_truth=truth,
                                         gradient=gradient, presentation=presentation),
                            config=PLOT_CONFIG, key="circle_image_chart")
        with right:
            st.plotly_chart(center_figure(dataset, active=active, visible=visible or len(dataset.points),
                                          radius=radius, phi=phi, show_loci=locus,
                                          intersections=intersections, grid=grid, votes=votes,
                                          layer=layer, show_peak=show_peak,
                                          show_candidate=candidate, presentation=presentation),
                            config=PLOT_CONFIG, key="circle_center_chart")


def _candidate_caption(dataset, active, radius, phi):
    point = dataset.points[active].xy
    center = candidate_center(point, radius, phi)
    st.caption(f"{dataset.points[active].id}=({point[0]:.3f}, {point[1]:.3f}) · "
               f"φ={phi:.1f}° · r={radius:.2f} → candidate center "
               f"({center[0]:.4f}, {center[1]:.4f}); khoảng cách tới point={np.linalg.norm(center-point):.4f}.")


def _vote_options():
    mode_label = st.session_state.circle_lab_voting
    mode = "full" if mode_label == "Full locus" else "unknown" if mode_label == "Unknown polarity" else "gradient"
    polarity = "outward" if st.session_state.circle_lab_polarity == "Outward" else "inward"
    return mode, polarity


def _grid():
    s = st.session_state
    width = float(s.circle_lab_radius_width)
    first_center = 1.0 if width >= 1 else .5
    radius_range = (first_center-width/2, 3.5+width/2)
    return make_grid(center_width=float(s.circle_lab_center_width),
                     radius=float(s.circle_lab_radius),
                     radius_range=radius_range if s.circle_lab_search == "Unknown radius" else None,
                     radius_width=width)


def _sync_votes(dataset, grid, mode, polarity):
    s = st.session_state
    signature = vote_signature(dataset, grid, mode, int(s.circle_lab_samples), polarity)
    if signature != s.circle_lab_signature or s.circle_lab_votes is None:
        s.circle_lab_votes = empty_votes(dataset, grid, mode, int(s.circle_lab_samples), polarity)
        s.circle_lab_signature = signature
        s.circle_lab_layer = 0
        return True
    return False


def _lab_vote(action):
    s = st.session_state
    dataset = s.circle_lab_dataset
    grid = _grid()
    mode, polarity = _vote_options()
    _sync_votes(dataset, grid, mode, polarity)
    state = s.circle_lab_votes
    index = int(s.circle_lab_point)
    kwargs = dict(mode=mode, angular_samples=int(s.circle_lab_samples), polarity=polarity)
    if action == "next":
        state = vote_next(state, dataset, grid, index, **kwargs)
    elif action == "previous":
        state = vote_previous(state, dataset, grid, index, **kwargs)
    elif action == "point":
        state = vote_remaining(state, dataset, grid, index, **kwargs)
    elif action == "all":
        state = vote_all(state, dataset, grid, **kwargs)
    elif action == "reset":
        state = empty_votes(dataset, grid, **kwargs)
    s.circle_lab_votes = state
    if action == "next" and state.last_event is not None:
        s.circle_lab_layer = state.last_event.radius_index


def _next_point():
    s = st.session_state
    s.circle_lab_point = (int(s.circle_lab_point) + 1) % len(s.circle_lab_dataset.points)
    s.circle_lab_visible = max(int(s.circle_lab_visible), int(s.circle_lab_point)+1)
    if s.circle_lab_votes is not None:
        s.circle_lab_votes = replace(s.circle_lab_votes, last_event=None)


def _guided_vote(action):
    dataset = generate_circle_example()
    grid = make_grid()
    state = st.session_state.circle_guided_votes
    if state is None:
        state = empty_votes(dataset, grid)
    if action == "next":
        state = vote_next(state, dataset, grid, 0)
    elif action == "point":
        state = vote_remaining(state, dataset, grid, 0)
    elif action == "all":
        state = vote_all(state, dataset, grid)
    elif action == "reset":
        state = empty_votes(dataset, grid)
    st.session_state.circle_guided_votes = state


def _load_preset():
    s = st.session_state
    name = s.circle_lab_preset
    s.circle_lab_dataset = preset_dataset(name)
    s.circle_loaded_preset = name
    s.circle_lab_point = 0
    s.circle_lab_visible = 1
    s.circle_lab_radius = 1.5 if name == PRESETS[1] else 2.0
    s.circle_lab_phi = -90.0
    s.pop("circle_lab_phi_control", None)
    s.pop("circle_lab_radius_control", None)
    s.circle_lab_search = "Unknown radius" if name == PRESETS[1] or name == PRESETS[5] else "Known radius"
    s.circle_lab_voting = "Full locus"
    s.circle_lab_polarity = "Outward" if name == PRESETS[4] else "Inward"
    s.circle_lab_votes = None
    s.circle_lab_signature = None
    s.circle_lab_prediction = None


def _add_custom():
    s = st.session_state
    dataset = s.circle_lab_dataset
    s.circle_custom_serial += 1
    gradient = None
    if s.get("circle_custom_gradient", False):
        gradient = (float(s.circle_add_gx), float(s.circle_add_gy))
    point = Observation(f"custom-{s.circle_custom_serial}",
                        (float(s.circle_add_x), float(s.circle_add_y)), gradient, False)
    s.circle_lab_dataset = replace(dataset, points=dataset.points + (point,))
    s.circle_lab_point = len(dataset.points)
    s.circle_lab_visible = len(dataset.points) + 1
    s.circle_lab_votes = None


def _remove_custom():
    s = st.session_state
    dataset = s.circle_lab_dataset
    selected = int(s.circle_lab_point)
    if not dataset.points[selected].reference:
        s.circle_lab_dataset = replace(dataset,
                                        points=tuple(p for p in dataset.points if p.id != dataset.points[selected].id))
        s.circle_lab_point = min(selected, len(s.circle_lab_dataset.points)-1)
        s.circle_lab_visible = min(s.circle_lab_visible, len(s.circle_lab_dataset.points))
        s.circle_lab_votes = None


def _vote_summary(dataset, grid, state, layer, *, show_truth=True):
    maximum, winners = peaks(state.counts)
    done = sum(state.next_indices)
    events = vote_events(dataset, grid, mode=_vote_options()[0],
                         angular_samples=int(st.session_state.circle_lab_samples),
                         polarity=_vote_options()[1])
    total = sum(map(len, events))
    st.caption(f"Đã thăm {done}/{total} candidate samples · "
               f"{int(state.counts.sum())} support votes hợp lệ · peak hiện tại {maximum}. "
               "Mỗi point chỉ góp tối đa 1 support/bin/radius; sample lặp ô không cộng thêm.")
    if done < total:
        st.caption("Peak còn TẠM THỜI; các point/chồng radius chưa vote xong.")
    if maximum:
        st.caption(f"{len(winners)} peak đồng hạng ở mức {maximum}. "
                   "Chọn một cell bên dưới để xem giả thuyết; không tự tuyên bố tâm duy nhất.")
    ranked = top_cells(state.counts, 10)
    detected = None
    selected_cell = None
    if ranked:
        options = list(range(len(ranked)))
        st.selectbox("Inspect candidate bin", options, key="circle_lab_detected",
                     format_func=lambda i: f"#{i+1} · {ranked[i][0]} support · "
                     f"(a,b,r)=({cell_parameters(grid, ranked[i][1])[0]:.2f}, "
                     f"{cell_parameters(grid, ranked[i][1])[1]:.2f}, "
                     f"{cell_parameters(grid, ranked[i][1])[2]:.2f})")
        selected = min(int(st.session_state.circle_lab_detected), len(ranked)-1)
        selected_cell = ranked[selected][1]
        detected = cell_parameters(grid, selected_cell)
        if show_truth and dataset.true_circles:
            errors = [(float(np.linalg.norm(np.array(detected[:2])-circle[:2])), abs(detected[2]-circle[2]))
                      for circle in dataset.true_circles]
            closest = min(errors)
            st.caption(f"Bin-center estimate: {detected} · gần circle nguồn nhất: "
                       f"sai số tâm {closest[0]:.3f}, bán kính {closest[1]:.3f}. "
                       "Đây là đánh giá sau vote, không dùng truth để chọn bin.")
    return detected, selected_cell


def guided_tutorial():
    _init()
    _header("Hough Circle Transform · Guided Tutorial",
            "Một edge point mở ra cả locus tâm; r, gradient và vote quyết định giả thuyết circle.")
    s = st.session_state
    step = int(s.circle_guided_step)
    _nav(step)
    _progress(step)
    st.subheader(f"Bước {step}/14 · {STEPS[step-1]}")
    dataset = generate_circle_example()
    radius = 2.0
    phi = float(s.circle_guided_phi)
    visible = 3 if step == 1 else 1 if step <= 3 else 2 if step == 4 else 3 if step == 5 else 6
    intersections = ()
    if step == 4 and s.circle_guided_reveal_4:
        intersections = common_centers([p.xy for p in dataset.points[:2]], radius)
    elif step in (5, 6):
        intersections = common_centers([p.xy for p in dataset.points[:visible]], radius)
    grid = make_grid()
    if s.circle_guided_votes is None:
        s.circle_guided_votes = empty_votes(dataset, grid)
    votes = s.circle_guided_votes
    detected = None
    if step in (2, 3, 7, 8):
        a, b = st.columns(2)
        with a:
            _float_slider("φ · góc candidate (°)", -180.0, 180.0, 5.0, "circle_guided_phi")
        if step >= 7:
            with b:
                _float_slider("r · radius hypothesis", .5, 3.5, .1, "circle_guided_radius")
            radius = float(s.circle_guided_radius)
        phi = float(s.circle_guided_phi)
    if step == 4:
        st.markdown('<div class="question">Dự đoán: hai điểm và r=2 còn bao nhiêu tâm chung?</div>', unsafe_allow_html=True)
        st.radio("Số nghiệm", ("0", "1", "2", "Vô số"), index=None, key="circle_guided_predict_4", horizontal=True)
        st.button("Reveal giao điểm", key="circle_reveal_4",
                  disabled=not s.get("circle_guided_predict_4"),
                  on_click=lambda: s.__setitem__("circle_guided_reveal_4", True))
    if step == 11:
        st.markdown('<div class="question">Gradient hướng vào có giúp giảm số candidate trên mỗi radius?</div>', unsafe_allow_html=True)
        st.radio("Dự đoán", ("Có", "Không", "Chưa chắc"), index=None,
                 key="circle_guided_predict_11", horizontal=True)
        st.button("Reveal gradient vote", key="circle_reveal_11",
                  disabled=not s.get("circle_guided_predict_11"),
                  on_click=lambda: s.__setitem__("circle_guided_reveal_11", True))
    if step in (9, 10):
        with st.container(key="circle_vote_actions"):
            a, b, c, d = st.columns(4)
            a.button("Next Vote", key="circle_guided_vote_next", on_click=_guided_vote, args=("next",), width="stretch")
            b.button("Vote Selected Point", key="circle_guided_vote_point", on_click=_guided_vote, args=("point",), width="stretch")
            c.button("Vote All", key="circle_guided_vote_all", on_click=_guided_vote, args=("all",), width="stretch")
            d.button("Reset Votes", key="circle_guided_vote_reset", on_click=_guided_vote, args=("reset",), width="stretch")
        votes = s.circle_guided_votes
        if votes.last_event:
            event = votes.last_event
            phi = event.sample_index * 360/72
            st.caption(f"p{event.point_index+1} · sample {event.sample_index+1}/72 · "
                       f"candidate=({event.candidate[0]:.3f},{event.candidate[1]:.3f}) · "
                       f"bin={event.cell} · {'+1 support' if event.counted else 'không cộng trùng/ngoài lưới'}.")
    if step == 13:
        detected_state = vote_all(empty_votes(dataset, grid, "gradient"), dataset, grid, mode="gradient")
        maximum, cells = peaks(detected_state.counts)
        detected = cell_parameters(grid, cells[0]) if len(cells) == 1 else None
        votes = detected_state
        st.caption(f"Gradient inward · {int(votes.counts.sum())} vote · max={maximum} · "
                   f"{len(cells)} ô đồng hạng. Detected = bin center, không thay bằng tâm nguồn.")
    _panels(dataset, active=0, visible=visible, radius=radius, phi=phi,
            locus=step >= 3, intersections=intersections,
            grid=grid if step in (9, 10, 13) else None,
            votes=votes if step in (9, 10, 13) else None,
            show_peak=step in (10, 13), detected=detected,
            truth=step in (5, 6, 13, 14), gradient=step in (11, 12, 13),
            candidate=step not in (1, 13), image_only=step == 1)
    if step in (2, 3, 7, 8):
        _candidate_caption(dataset, 0, radius, phi)
    if step == 1:
        st.info("Ba edge point xanh trên vòng tròn nguồn. Chỉ nhìn một point thì chưa biết tâm nào là đúng.")
    elif step == 2:
        st.info("Một point + r cố định vẫn cho vô số candidate circle. Thử quét φ: circle vàng luôn đi qua p1.")
    elif step == 3:
        st.latex(r"(a-x)^2+(b-y)^2=r^2,\quad (a,b)=p+r(\cos\phi,\sin\phi)")
        st.caption("Một edge point trong Image Space biến thành một vòng tròn tâm khả dĩ trong Center Space.")
    elif step == 4 and s.circle_guided_reveal_4:
        st.success(f"Có {len(intersections)} tâm liên tục hợp lệ: " + "; ".join(f"({p[0]:.3f},{p[1]:.3f})" for p in intersections) + ". Chưa thể chọn tâm thật chỉ từ hai điểm.")
    elif step == 5:
        st.success(f"Điểm thứ ba giữ lại {len(intersections)} nghiệm chung: ({intersections[0][0]:.3f}, {intersections[0][1]:.3f}). Không dùng nhãn truth để chọn nghiệm.")
    elif step == 6:
        st.caption("Sáu locus lý tưởng cắt nhau tại cùng tâm. Đây là bằng chứng hình học liên tục, chưa phải accumulator.")
    elif step in (7, 8):
        st.caption("Mỗi r tạo một locus khác. Các lớp r=1, 2, 3 đều tính bằng p+r(cosφ,sinφ). "
                   "Line Hough thường 2 tham số; circle thêm chiều r → tìm kiếm và bộ nhớ tăng.")
        if step == 8:
            st.dataframe([{"r": layer, "candidate a": candidate_center(dataset.points[0].xy, layer, phi)[0],
                           "candidate b": candidate_center(dataset.points[0].xy, layer, phi)[1]}
                          for layer in (1., 2., 3.)], hide_index=True, width="stretch")
    elif step in (9, 10):
        maximum, cells = peaks(votes.counts)
        st.caption(f"Grid A[r,b_row,a_col] · Δa=Δb=.4 · "
                   f"{sum(votes.next_indices)} angular samples đã thăm → {votes.counts.sum()} support votes. "
                   f"Peak tạm={maximum}, {len(cells)} ô đồng hạng. Một point/bin/layer tối đa một vote.")
    elif step == 11 and s.circle_guided_reveal_11:
        obs = dataset.points[0]
        center = gradient_candidates(obs.xy, 2, obs.gradient, "inward")[0]
        st.success(f"Bright inside → gradient hướng vào. p1 + 2·ĝ = ({center[0]:.3f},{center[1]:.3f}). "
                   "Full locus có 72 angular samples; gradient hợp lệ chỉ tạo 1 candidate/radius.")
    elif step == 12:
        obs = dataset.points[0]
        both = gradient_candidates(obs.xy, 2, obs.gradient, "unknown")
        st.caption(f"Chưa biết polarity: p ± r·ĝ → ({both[0][0]:.2f},{both[0][1]:.2f}) và "
                   f"({both[1][0]:.2f},{both[1][1]:.2f}). Bright outside làm gradient đảo hướng. "
                   "Gradient zero hoặc không tin cậy không tạo vote định hướng.")
    elif step == 13 and detected:
        st.success(f"Detected bin center = ({detected[0]:.3f},{detected[1]:.3f}), "
                   f"r={detected[2]:.3f}; sai số tâm={np.linalg.norm(np.array(detected[:2])-CENTER):.3f}. "
                   "Vòng xanh là kết quả vote, xám nét đứt là nguồn.")
    elif step == 14:
        st.info("Sai r có thể loại tâm thật; bin quá thô, sample góc thưa, cung thiếu, outlier, "
                "hai circle, polarity sai hoặc gradient zero đều ảnh hưởng peak. "
                "Peak là bằng chứng trên lưới, không bảo đảm vật thể đúng.")
        st.markdown("**Edge points → Candidate centers → Radius hypotheses → Voting → Peak → Detected circle**")


def experiment_lab():
    _init()
    _header("Hough Circle Transform · Experiment Lab",
            "Đồng bộ Image Space và Center Space; thử giao loci, rồi cast từng vote trên lưới thật.")
    s = st.session_state
    st.selectbox("Teaching preset", PRESETS, key="circle_lab_preset", on_change=_load_preset)
    dataset = s.circle_lab_dataset
    st.markdown(f'<div class="question"><b>Dự đoán:</b> {dataset.question}</div>', unsafe_allow_html=True)
    st.radio("Dự đoán trước khi vote", ("Một peak rõ", "Nhiều peak / mơ hồ", "Phụ thuộc tham số"),
             index=None, key="circle_lab_prediction", horizontal=True)
    st.radio("Workspace", ("A · Center Locus", "B · Circle Voting"), key="circle_lab_workspace", horizontal=True)
    _point_picker(dataset, "circle_lab_point")
    active = int(s.circle_lab_point)
    if s.circle_lab_workspace.startswith("A"):
        with st.container(key="circle_primary_controls"):
            a, b, c = st.columns(3)
            with a:
                _float_slider("φ · candidate angle (°)", -180.0, 180.0, 5.0, "circle_lab_phi")
            with b:
                _float_slider("r · geometric radius", .5, 3.5, .1, "circle_lab_radius")
            c.slider("Visible loci", 1, len(dataset.points), key="circle_lab_visible")
    else:
        _float_slider("r · geometric radius", .5, 3.5, .1, "circle_lab_radius",
                      disabled=s.circle_lab_search == "Unknown radius")
        st.caption("Candidate circle sau Next Vote theo đúng sample vừa cast; đổi φ/loci tại workspace A.")
    radius, phi = float(s.circle_lab_radius), float(s.circle_lab_phi)
    if s.circle_lab_workspace.startswith("A"):
        st.toggle("Hiện toàn bộ center loci", key="circle_lab_show_locus")
        visible_points = [p.xy for p in dataset.points[:int(s.circle_lab_visible)]]
        try:
            intersections = common_centers(visible_points, radius) if len(visible_points) >= 2 else ()
            message = f"{len(intersections)} giao điểm hợp lệ cho {len(visible_points)} point ở r={radius:.2f}."
        except ValueError as error:
            intersections, message = (), str(error)
        _panels(dataset, active=active, visible=int(s.circle_lab_visible), radius=radius, phi=phi,
                locus=bool(s.circle_lab_show_locus), intersections=intersections)
        _candidate_caption(dataset, active, radius, phi)
        st.caption(message)
        with st.expander("Thêm/bỏ custom observation"):
            a, b = st.columns(2)
            a.number_input("x custom", -10.0, 10.0, value=0.0, step=.1, key="circle_add_x")
            b.number_input("y custom", -10.0, 10.0, value=0.0, step=.1, key="circle_add_y")
            st.toggle("Tự nhập gradient (không suy từ tâm nguồn)", key="circle_custom_gradient")
            if s.circle_custom_gradient:
                a, b = st.columns(2)
                a.number_input("gx custom", -10.0, 10.0, value=0.0, step=.1, key="circle_add_gx")
                b.number_input("gy custom", -10.0, 10.0, value=0.0, step=.1, key="circle_add_gy")
            a, b = st.columns(2)
            a.button("Add custom point", key="circle_add_point", on_click=_add_custom, disabled=len(dataset.points) >= 30)
            b.button("Remove selected custom point", key="circle_remove_point", on_click=_remove_custom,
                     disabled=dataset.points[active].reference)
            st.caption("Điểm nguồn không bị xóa. Custom point thiếu gradient không có gradient-informed vote.")
    else:
        with st.expander("Voting setup · search, polarity, bin và sampling"):
            with st.container(key="circle_voting_controls"):
                a, b, c = st.columns(3)
                a.radio("Search", ("Known radius", "Unknown radius"), key="circle_lab_search")
                b.radio("Voting", ("Full locus", "Known gradient", "Unknown polarity"), key="circle_lab_voting")
                c.radio("Gradient polarity", ("Inward", "Outward"), key="circle_lab_polarity",
                        disabled=s.circle_lab_voting != "Known gradient")
                a, b, c = st.columns(3)
                a.select_slider("Center bin width", (.2, .4, .5, 1.0, 2.0), value=.4,
                                key="circle_lab_center_width")
                b.select_slider("Radius bin width", (.25, .5, 1.0), value=.5,
                                key="circle_lab_radius_width")
                c.select_slider("Angular samples", (12, 24, 36, 72, 120, 180), value=72,
                                key="circle_lab_samples")
        try:
            grid = _grid()
            mode, polarity = _vote_options()
            invalidated = _sync_votes(dataset, grid, mode, polarity)
            events = vote_events(dataset, grid, mode=mode,
                                 angular_samples=int(s.circle_lab_samples), polarity=polarity)
        except ValueError as error:
            st.error(f"Cấu hình không hợp lệ: {error}. Giảm số bin/lớp hoặc chỉnh bounds.")
            return
        if invalidated:
            st.caption("Dataset/radius/bin/sampling/polarity đã đổi: accumulator cũ được reset.")
        if s.circle_lab_search == "Unknown radius":
            st.select_slider("Radius layer · bin center", options=list(range(grid.shape[0])),
                             format_func=lambda i: f"r={grid.r_centers[i]:.3f}", key="circle_lab_layer")
        layer = min(int(s.circle_lab_layer), grid.shape[0]-1)
        display_radius = float(grid.r_centers[layer])
        if s.circle_lab_search == "Unknown radius":
            st.caption(f"Đang xem radius slice r={display_radius:.3f}; candidate circle/locus và heatmap cùng dùng lớp này.")
        progress = int(s.circle_lab_votes.next_indices[active])
        st.caption(f"Point {dataset.points[active].id}: {progress}/{len(events[active])} candidate samples."
                   + (" Gradient thiếu/zero: không thể tạo vote định hướng cho point này."
                      if mode != "full" and not events[active] else ""))
        with st.container(key="circle_vote_actions"):
            columns = st.columns(6)
            for col, (label, action) in zip(columns, (("← Previous Step", "previous"),
                                                       ("Next Vote", "next"), ("Next Point", "next_point"),
                                                       ("Vote Selected Point", "point"),
                                                       ("Vote All", "all"), ("Reset Votes", "reset"))):
                if action == "next_point":
                    col.button(label, key="circle_vote_next_point", width="stretch", on_click=_next_point)
                else:
                    col.button(label, key=f"circle_vote_{action}", width="stretch",
                               disabled=(action == "next" and progress >= len(events[active])) or
                                        (action == "previous" and progress == 0),
                               on_click=_lab_vote, args=(action,))
        state = s.circle_lab_votes
        if state.last_event:
            event = state.last_event
            st.caption(f"Sample thật: {dataset.points[event.point_index].id} · radius layer {event.radius_index+1} · "
                       f"candidate=({event.candidate[0]:.3f},{event.candidate[1]:.3f}) → "
                       f"A{event.cell} · {'+1 support' if event.counted else 'đã thăm bin này / ngoài lưới'}.")
        visual_slot = st.container()
        detected, selected_cell = _vote_summary(dataset, grid, state, layer)
        event = state.last_event
        if event is not None and event.radius_index == layer:
            visual_active = event.point_index
            visual_radius = float(grid.r_centers[event.radius_index])
            source = np.asarray(dataset.points[visual_active].xy)
            visual_phi = float(np.rad2deg(np.arctan2(event.candidate[1]-source[1],
                                                        event.candidate[0]-source[0])))
        else:
            visual_active, visual_radius, visual_phi = active, display_radius, phi
        with visual_slot:
            if selected_cell is not None and selected_cell[0] != layer:
                st.caption(f"Circle xanh lấy từ detected bin ở r={grid.r_centers[selected_cell[0]]:.3f}; "
                           f"heatmap bên phải đang hiển thị radius slice r={grid.r_centers[layer]:.3f}.")
            _panels(dataset, active=visual_active, visible=len(dataset.points),
                    radius=visual_radius, phi=visual_phi, locus=False, grid=grid,
                    votes=state, layer=layer, show_peak=True, detected=detected,
                    truth=bool(s.circle_lab_prediction), gradient=mode != "full")
        if s.circle_lab_search == "Unknown radius":
            summaries = [{"r layer": round(float(r), 3), "max support": int(state.counts[i].max(initial=0)),
                          "support votes": int(state.counts[i].sum())}
                         for i, r in enumerate(grid.r_centers)]
            st.dataframe(summaries, hide_index=True, width="stretch")
            st.caption("A[r_index,b_row,a_col] · các lớp chia chung center grid; r là tâm bin, không phải nghiệm fit liên tục.")
        if s.circle_lab_prediction:
            st.info(f"Quan sát từ vote hiện tại: {dataset.limitation} "
                    "Một peak là bằng chứng trên discretization này, không bảo đảm object thật.")
