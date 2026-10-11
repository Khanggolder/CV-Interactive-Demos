"""Guided LSD teaching prototype and two connected experiment workspaces."""

from __future__ import annotations

from math import degrees, pi

import numpy as np
import streamlit as st

from utils.lsd import (DEFAULT_EPSILON, DEFAULT_MIN_MAGNITUDE, DEFAULT_TAU,
                       PRESETS, PRESET_NOTES, analyze_field, angular_difference_2pi,
                       angular_difference_pi, format_probability, generate_lsd_reference_field,
                       fit_lsr_segment, grow_lsr, preset_field, replay_lsr_events,
                       select_lsd_seed)
from utils.lsd_visualization import field_figure
from utils.visualization import PLOT_CONFIG

STEPS = (
    "Từ edge đến đoạn hữu hạn", "Gradient đo được", "Level-line vuông góc",
    "Chọn seed mạnh nhất", "Axial ≠ directed", "So sánh neighbor",
    "Hoàn tất LSR", "Weighted PCA", "Support rectangle", "Vì sao cần NFA?",
    "Candidate A · kiểm định", "Candidate B · kiểm định", "Toàn bộ candidates",
    "Giới hạn và tổng kết",
)


@st.cache_data(show_spinner=False)
def _analyze(preset, tau_deg, min_magnitude, epsilon):
    return analyze_field(preset_field(preset), np.deg2rad(tau_deg), min_magnitude, epsilon)


def _init():
    defaults = {
        "lsd_guided_step": 1, "lsd_guided_pixel_y": 6, "lsd_guided_pixel_x": 8,
        "lsd_guided_reveal_seed": False, "lsd_guided_replay": 0,
        "lsd_guided_11_calc": False, "lsd_guided_12_calc": False,
        "lsd_guided_11_decision": False, "lsd_guided_12_decision": False,
        "lsd_lab_preset": PRESETS[0], "lsd_lab_workspace": "A · Gradient & Region",
        "lsd_lab_tau": 22.5, "lsd_lab_min": .09, "lsd_lab_epsilon": 1.0,
        "lsd_lab_pixel_y": 6, "lsd_lab_pixel_x": 8,
        "lsd_lab_view": "Intensity", "lsd_lab_seed_mode": "Strongest eligible",
        "lsd_lab_started": False, "lsd_lab_replay": 0, "lsd_lab_candidate": 0,
        "lsd_lab_nfa_calc": False, "lsd_lab_nfa_decision": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _header(title, subtitle):
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("lsd_presentation", False):
        st.markdown(f'<div class="hero-copy">{subtitle}</div>', unsafe_allow_html=True)


def _progress(step):
    labels = ("Input", "∇I", "Level", "Seed", "Angles", "BFS", "LSR", "PCA",
              "Box", "NFA", "A", "B", "All", "Limits")
    cells = "".join(f'<div class="step {"active" if i == step else "done" if i < step else ""}">{i}<br>{label}</div>'
                    for i, label in enumerate(labels, 1))
    st.markdown(f'<div class="stepper lsd-stepper">{cells}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-progress">Tiến độ học · {step}/14'
                f'<div class="progress-track"><span style="width:{step/14*100:.1f}%"></span></div></div>',
                unsafe_allow_html=True)


def _reset_guided():
    s = st.session_state
    s.lsd_guided_step = 1
    s.lsd_guided_replay = 0
    s.lsd_guided_reveal_seed = False
    for step in (11, 12):
        s[f"lsd_guided_{step}_calc"] = False
        s[f"lsd_guided_{step}_decision"] = False
        s.pop(f"lsd_guided_{step}_predict", None)
    s.pop("lsd_guided_seed_predict", None)


def _move_step(delta):
    s = st.session_state
    s.lsd_guided_step = min(14, max(1, s.lsd_guided_step + delta))


def _guided_gate(step, event_count):
    s = st.session_state
    return ((step == 4 and not s.lsd_guided_reveal_seed)
            or (step == 6 and s.lsd_guided_replay == 0 and event_count > 0)
            or (step == 7 and s.lsd_guided_replay < event_count)
            or (step in (11, 12) and not s[f"lsd_guided_{step}_decision"]))


def _nav(step, event_count):
    with st.container(key="lsd_nav"):
        a, b, c = st.columns(3)
        a.button("← Previous", key="lsd_previous", disabled=step == 1,
                 on_click=_move_step, args=(-1,), width="stretch")
        b.button("Next →", key="lsd_next", disabled=step == 14 or _guided_gate(step, event_count),
                 on_click=_move_step, args=(1,), width="stretch")
        c.button("Reset", key="lsd_reset", on_click=_reset_guided, width="stretch")


def _chart(data, *, key, **options):
    options.setdefault("presentation", bool(st.session_state.get("lsd_presentation", False)))
    st.plotly_chart(field_figure(data, **options), config=PLOT_CONFIG, key=key)


def _pixel_inspector(data, prefix):
    h, w = data["intensity"].shape
    s = st.session_state
    s[f"{prefix}_pixel_y"] = min(int(s[f"{prefix}_pixel_y"]), h-1)
    s[f"{prefix}_pixel_x"] = min(int(s[f"{prefix}_pixel_x"]), w-1)
    on_change = (_invalidate_lab if prefix == "lsd_lab" and
                 s.get("lsd_lab_seed_mode") == "Manual exploration" else None)
    a, b = st.columns(2)
    with a:
        st.selectbox("y · hàng ↑", range(h), key=f"{prefix}_pixel_y", on_change=on_change)
    with b:
        st.selectbox("x · cột →", range(w), key=f"{prefix}_pixel_x", on_change=on_change)
    pixel = (int(s[f"{prefix}_pixel_y"]), int(s[f"{prefix}_pixel_x"]))
    y, x = pixel
    mag = data["magnitude"][pixel]
    st.caption(f"(y,x)=({y},{x}) · I={data['intensity'][pixel]:.4f} · "
               f"Gx={data['gx'][pixel]:+.4f}, Gy={data['gy'][pixel]:+.4f} · M={mag:.4f} · "
               + (f"θg={degrees(np.arctan2(data['gy'][pixel], data['gx'][pixel])):.1f}°, "
                  f"θlevel={degrees(data['angles'][pixel]):.1f}°" if mag > 1e-12 else
                  "gradient/level-line direction: undefined (M=0)"))
    return pixel


def _replay_controls(region, key):
    s = st.session_state
    count = len(region["events"])
    s[key] = min(int(s[key]), count)
    with st.container(key="lsd_replay_actions"):
        cols = st.columns(3)
        cols[0].button("Start Region", key=f"{key}_start", on_click=lambda: s.__setitem__(key, 0), width="stretch")
        cols[1].button("Next Neighbor", key=f"{key}_next", disabled=s[key] >= count,
                       on_click=lambda: s.__setitem__(key, min(count, s[key]+1)), width="stretch")
        cols[2].button("Previous Neighbor", key=f"{key}_previous", disabled=s[key] == 0,
                       on_click=lambda: s.__setitem__(key, max(0, s[key]-1)), width="stretch")
        cols = st.columns(3)
        cols[0].button("Run Several Steps", key=f"{key}_batch", disabled=s[key] >= count,
                       on_click=lambda: s.__setitem__(key, min(count, s[key]+5)), width="stretch")
        cols[1].button("Complete Region", key=f"{key}_complete", disabled=s[key] >= count,
                       on_click=lambda: s.__setitem__(key, count), width="stretch")
        cols[2].button("Reset Region", key=f"{key}_reset", on_click=lambda: s.__setitem__(key, 0), width="stretch")
    replay = replay_lsr_events(region, int(s[key]))
    st.caption(f"Comparison {replay['count']}/{count} · thành viên hiện có {len(replay['members'])}/{len(region['pixels'])}. "
               "Pixel đã visited/used hoặc dưới ngưỡng được BFS bỏ qua trước phép so góc.")
    if replay["current"] is not None:
        e = replay["current"]
        st.info(f"Parent {e['parent']} → neighbor {e['pixel']} · M={e['magnitude']:.3f} · "
                f"θ={degrees(e['angle']):.1f}° · seed θ={degrees(e['reference']):.1f}° · "
                f"Δaxial={degrees(e['difference']):.1f}° {'<' if e['accepted'] else '≥'} "
                f"τ={degrees(e['tau']):.1f}° → {'thêm vào LSR' if e['accepted'] else 'loại khỏi LSR'}.")
    return replay


def _fit_metrics(region):
    fit = region["fit"]
    a, b, c = st.columns(3)
    a.metric("LSR pixels · n", fit["n"])
    b.metric("Centroid (x,y)", f"({fit['center'][0]:.2f}, {fit['center'][1]:.2f})")
    c.metric("Tangent angle", f"{degrees(fit['angle']):.1f}°")
    st.caption(f"Đoạn hữu hạn từ ({fit['endpoints'][0,0]:.2f}, {fit['endpoints'][0,1]:.2f}) "
               f"đến ({fit['endpoints'][1,0]:.2f}, {fit['endpoints'][1,1]:.2f}); "
               f"rectangle dài {fit['length']:.2f}, rộng {fit['width']:.2f} pixel.")


def _nfa_microscope(data, region, prefix, *, gate=True):
    s = st.session_state
    fit = region["fit"]
    st.markdown("**Region growing:** so hướng *axial* modulo π với seed cố định. "
                "**NFA:** so hướng *directed* modulo 2π với segment đã fit; pixel cùng trục nhưng ngược polarity có thể không aligned.")
    a, b, c = st.columns(3)
    a.metric("n", fit["n"])
    b.metric("k", fit["k"])
    c.metric("n−k", fit["n"]-fit["k"])
    st.caption("n = số pixel trong LSR; k = aligned theo hướng directed của segment; n−k = misaligned.")
    st.caption(f"τ={degrees(data['tau']):.1f}° · p₀=τ/π={data['tau']/pi:.4f} · "
               f"N_T={data['tests']} grown LSRs · ε={data['epsilon']:.3g}. "
               "Cyan = aligned; cam = misaligned.")
    st.radio("Dự đoán quyết định", ("ACCEPT", "REJECT"), index=None,
             key=f"{prefix}_predict", horizontal=True)
    st.button("Reveal phép tính binomial", key=f"{prefix}_calc_button",
              disabled=gate and not s.get(f"{prefix}_predict"),
              on_click=lambda: s.__setitem__(f"{prefix}_calc", True))
    if s.get(f"{prefix}_calc", False):
        st.latex(r"P(X\geq k)=\sum_{j=k}^{n}\binom{n}{j}p_0^j(1-p_0)^{n-j}")
        st.latex(r"\mathrm{NFA}=N_T\,P(X\geq k)")
        a, b = st.columns(2)
        a.metric(f"P(X≥{fit['k']})", format_probability(region["log_tail"]))
        b.metric("NFA", format_probability(region["log_nfa"]))
        st.button("Reveal ACCEPT / REJECT", key=f"{prefix}_decision_button",
                  on_click=lambda: s.__setitem__(f"{prefix}_decision", True))
    if s.get(f"{prefix}_decision", False):
        decision = "ACCEPT" if region["accept"] else "REJECT"
        (st.success if region["accept"] else st.warning)(
            f"{decision}: NFA {format_probability(region['log_nfa'])} "
            f"{'<' if region['accept'] else '≥'} ε {data['epsilon']:.3g}. "
            "NFA nhỏ hơn = bằng chứng mạnh hơn dưới giả định của mô hình này; "
            "không chứng minh có biên vật thể thật.")
    st.caption("NFA này chỉ dùng các LSR được chọn thích nghi và N_T là số vùng đã grow, "
               "không phải mọi rectangle-test của LSD gốc; không có bảo đảm false alarm toàn ảnh đã hiệu chuẩn.")


def guided_tutorial():
    _init()
    s = st.session_state
    _header("LSD · Line Segment Detector", "Từ pixel đến đoạn hữu hạn: dự đoán, chạy BFS thật, fit PCA rồi kiểm định NFA.")
    data = _analyze(PRESETS[0], 22.5, .09, 1.0)
    region_a, region_b = data["regions"][:2]
    step = int(s.lsd_guided_step)
    _progress(step)
    st.subheader(f"Bước {step}/14 · {STEPS[step-1]}")
    if step == 1:
        st.markdown('<div class="question">Hough Line cho supporting line kéo dài vô hạn. Ảnh này chứa đoạn biên hữu hạn ở đâu?</div>', unsafe_allow_html=True)
        _chart(data, key="lsd_guided_chart", view="Intensity")
        st.caption("10 hàng × 12 cột, [y,x] zero-based và y tăng lên; màu được chuẩn hóa chỉ để hiển thị, I thô không bị clip.")
    elif step in (2, 3):
        pixel = _pixel_inspector(data, "lsd_guided")
        _chart(data, key="lsd_guided_chart", view="Gradient arrows" if step == 2 else "Level lines", selected=pixel)
        st.info("∇I=(Gx,Gy), M=hypot(Gx,Gy)." if step == 2 else
                "θlevel=(atan2(Gy,Gx)−90°) mod 360°. Cyan vuông góc gradient tím; nếu M=0 thì hướng không xác định.")
    elif step == 4:
        seed = region_a["seed"]
        eligible = np.argwhere(data["magnitude"] >= data["min_magnitude"])
        top = sorted((tuple(map(int, p)) for p in eligible), key=lambda p: -data["magnitude"][p])[:6]
        st.selectbox("Dự đoán seed mạnh nhất (y,x)", top, index=None, key="lsd_guided_seed_predict",
                     format_func=lambda p: f"{p} · M={data['magnitude'][p]:.4f}")
        st.button("Reveal seed", key="lsd_reveal_seed", disabled=s.get("lsd_guided_seed_predict") is None,
                  on_click=lambda: s.__setitem__("lsd_guided_reveal_seed", True))
        _chart(data, key="lsd_guided_chart", view="Magnitude",
               region=region_a if s.lsd_guided_reveal_seed else None,
               replay=replay_lsr_events(region_a, 0) if s.lsd_guided_reveal_seed else None)
        if s.lsd_guided_reveal_seed:
            st.success(f"Seed = {seed}, M={data['magnitude'][seed]:.5f}, θlevel={degrees(data['angles'][seed]):.1f}°. "
                       "Quy tắc là gradient mạnh nhất chưa dùng; nhiễu/texture cũng có thể mạnh.")
    elif step == 5:
        st.slider("Chọn comparison thực của region A", 0, max(0, len(region_a["events"])-1), key="lsd_guided_angle_event")
        e = region_a["events"][s.lsd_guided_angle_event]
        directed = degrees(angular_difference_2pi(e["angle"], e["reference"]))
        st.info(f"Neighbor {e['pixel']} → seed {e['parent']}: Δaxial={degrees(e['difference']):.2f}°, "
                f"Δdirected={directed:.2f}°. Growth dùng Δaxial < τ; seed reference giữ cố định.")
        _chart(data, key="lsd_guided_chart", view="Level lines", region=region_a,
               replay=replay_lsr_events(region_a, s.lsd_guided_angle_event+1))
    elif step in (6, 7):
        replay = _replay_controls(region_a, "lsd_guided_replay")
        _chart(data, key="lsd_guided_chart", view="Intensity", region=region_a, replay=replay)
        if step == 7 and replay["count"] == len(region_a["events"]):
            st.success(f"LSR hoàn chỉnh có {len(region_a['pixels'])} pixel. Rebuild từ prefix event bằng đúng BFS final.")
    elif step in (8, 9):
        _chart(data, key="lsd_guided_chart", view="Intensity", region=region_a,
               fit=True, rectangle=step == 9)
        _fit_metrics(region_a)
        st.caption("Tâm = trung bình tọa độ có trọng số M; trục chính = eigenvector lớn nhất của scatter có trọng số. "
                   "Endpoints lấy từ projection nhỏ/lớn nhất của pixel support." if step == 8 else
                   "Padding theo nửa pixel-square trên tangent/normal, nên rectangle bao toàn bộ ô của LSR.")
    elif step == 10:
        st.markdown('<div class="question">Một đoạn fit đẹp có tự động đáng tin không?</div>', unsafe_allow_html=True)
        _chart(data, key="lsd_guided_chart", view="Intensity", region=region_a, fit=True, rectangle=True)
        st.info("Không. LSR n pixel được grow bằng axial angle; k pixel aligned với hướng đoạn theo directed angle. "
                "NFA kiểm tra mức hiếm của k dưới p₀=τ/π.")
    elif step in (11, 12):
        region = region_a if step == 11 else region_b
        st.caption(f"Candidate {'A' if step == 11 else 'B'} · seed {region['seed']} · "
                   "hãy dự đoán trước khi mở phép tính và kết luận.")
        if s.get("lsd_presentation", False):
            with st.container(key="lsd_nfa_panels"):
                left, right = st.columns([1.1, 1], gap="medium")
                with left:
                    _chart(data, key="lsd_guided_chart", view="Intensity", region=region,
                           fit=True, rectangle=True, nfa=True,
                           show_decision=bool(s[f"lsd_guided_{step}_decision"]))
                with right:
                    _nfa_microscope(data, region, f"lsd_guided_{step}")
        else:
            _chart(data, key="lsd_guided_chart", view="Intensity", region=region,
                   fit=True, rectangle=True, nfa=True,
                   show_decision=bool(s[f"lsd_guided_{step}_decision"]))
            _nfa_microscope(data, region, f"lsd_guided_{step}")
    elif step == 13:
        _chart(data, key="lsd_guided_chart", view="Intensity", all_segments=True)
        accepted = sum(r["accept"] for r in data["regions"])
        st.metric("Grown LSR candidates · N_T", data["tests"])
        st.caption(f"{accepted} accepted, {data['tests']-accepted} rejected. Xanh lá là đoạn hữu hạn được ACCEPT; "
                   "A và B chỉ là hai ví dụ trong toàn bộ phép quét.")
        with st.expander("Xem mọi candidate được tính"):
            st.dataframe([{"#": i+1, "seed (y,x)": r["seed"], "n": r["fit"]["n"],
                           "k": r["fit"]["k"], "NFA": format_probability(r["log_nfa"]),
                           "decision": "ACCEPT" if r["accept"] else "REJECT"}
                          for i, r in enumerate(data["regions"])], hide_index=True)
    else:
        _chart(data, key="lsd_guided_chart", view="Intensity", all_segments=True)
        st.markdown("**Intensity → Gradient → Level lines → Seed → 8-neighbor BFS → Weighted PCA → NFA → finite segments.**")
        st.info("Noise, τ, minimum magnitude và ε có thể đổi vùng/kết luận. Đây là LSD teaching prototype "
                "dựa trên LSR và binomial NFA, không phải LSD reference implementation đầy đủ; "
                "NFA không phải bảo đảm false alarm toàn ảnh đã hiệu chuẩn.")
        st.radio("Concept check: hai pixel ngược level-line polarity có thể cùng LSR nhưng khác NFA alignment?",
                 ("Có", "Không"), index=None, key="lsd_guided_check", horizontal=True)
        if s.get("lsd_guided_check"):
            st.success("Đúng: growth dùng modulo π, NFA dùng modulo 2π." if s.lsd_guided_check == "Có" else
                       "Hãy so lại axial (mod π) với directed (mod 2π).")
    _nav(step, len(region_a["events"]))


def _invalidate_lab():
    s = st.session_state
    s.lsd_lab_started = False
    s.lsd_lab_replay = 0
    s.lsd_lab_candidate = 0
    s.lsd_lab_nfa_calc = False
    s.lsd_lab_nfa_decision = False
    s.pop("lsd_lab_nfa_predict", None)


def _candidate_change():
    s = st.session_state
    s.lsd_lab_nfa_calc = False
    s.lsd_lab_nfa_decision = False
    s.pop("lsd_lab_nfa_predict", None)


def _reset_lab():
    s = st.session_state
    s.lsd_lab_preset = PRESETS[0]
    s.lsd_lab_tau = 22.5
    s.lsd_lab_min = .09
    s.lsd_lab_epsilon = 1.0
    s.lsd_lab_seed_mode = "Strongest eligible"
    s.lsd_lab_pixel_y, s.lsd_lab_pixel_x = 6, 8
    _invalidate_lab()
    s.pop("lsd_lab_prediction", None)


def experiment_lab():
    _init()
    s = st.session_state
    _header("LSD · Experiment Lab", "Thay intensity và ngưỡng; xem vùng BFS, đoạn PCA và NFA được tính lại trên cùng dữ liệu.")
    with st.expander("Dữ liệu và tham số · chỉnh trước khi quan sát", expanded=False):
        st.selectbox("Teaching preset", PRESETS, key="lsd_lab_preset", on_change=_invalidate_lab)
        st.caption(PRESET_NOTES[s.lsd_lab_preset][0])
        a, b, c = st.columns(3)
        a.slider("τ · angular tolerance (°)", 5.0, 75.0, step=2.5, key="lsd_lab_tau", on_change=_invalidate_lab)
        b.slider("Minimum gradient M", 0.0, 0.50, step=.01, key="lsd_lab_min", on_change=_invalidate_lab)
        c.slider("ε · NFA threshold", .01, 5.0, step=.01, key="lsd_lab_epsilon", on_change=_candidate_change)
        st.caption("Đổi τ/M tái tạo gradient eligibility, seed, mọi LSR, fit, k, p₀, N_T và NFA; đổi ε chỉ đổi quyết định. "
                   "Replay/reveal cũ được reset.")
    data = _analyze(s.lsd_lab_preset, s.lsd_lab_tau, s.lsd_lab_min, s.lsd_lab_epsilon)
    st.radio("Dự đoán trước khi khám phá", ("Ít candidate", "Nhiều candidate", "Chưa chắc"),
             index=None, key="lsd_lab_prediction", horizontal=True)
    st.radio("Workspace", ("A · Gradient & Region", "B · Segment & NFA"),
             key="lsd_lab_workspace", horizontal=True)
    if s.lsd_lab_workspace.startswith("A"):
        st.subheader("A · Gradient & Region Explorer")
        st.selectbox("Hiển thị", ("Intensity", "Magnitude", "Gradient arrows", "Level lines"), key="lsd_lab_view")
        h, w = data["intensity"].shape
        pixel = (min(int(s.lsd_lab_pixel_y), h-1), min(int(s.lsd_lab_pixel_x), w-1))
        seed = select_lsd_seed(data["magnitude"], np.zeros_like(data["magnitude"], dtype=bool), data["min_magnitude"])
        mode = s.lsd_lab_seed_mode
        if mode == "Manual exploration":
            seed = pixel if data["magnitude"][pixel] > 0 and data["magnitude"][pixel] >= data["min_magnitude"] else None
        region = None
        replay = None
        if seed is not None and s.lsd_lab_started:
            if mode == "Manual exploration":
                used = np.zeros_like(data["magnitude"], dtype=bool)
                region = grow_lsr(seed, data["angles"], data["magnitude"], used,
                                  data["tau"], data["min_magnitude"])
                region["fit"] = fit_lsr_segment(region, data["magnitude"], data["angles"], data["tau"])
            else:
                region = data["regions"][0]
            s.lsd_lab_replay = min(int(s.lsd_lab_replay), len(region["events"]))
            replay = replay_lsr_events(region, s.lsd_lab_replay)
        _chart(data, key="lsd_lab_chart", view=s.lsd_lab_view, selected=pixel,
               region=region, replay=replay,
               fit=bool(region and replay and replay["count"] == len(region["events"])),
               rectangle=bool(region and replay and replay["count"] == len(region["events"])))
        pixel = _pixel_inspector(data, "lsd_lab")
        mode = st.radio("Seed rule", ("Strongest eligible", "Manual exploration"),
                        key="lsd_lab_seed_mode", horizontal=True, on_change=_invalidate_lab)
        if mode == "Manual exploration":
            if seed is None:
                st.warning("Pixel thủ công dưới minimum magnitude: chọn pixel khác hoặc hạ ngưỡng.")
            st.caption("MANUAL EXPLORATION: chỉ grow một region; không gán N_T/NFA của full-image scan cho seed này.")
        if seed is None:
            st.warning("Không có seed đủ điều kiện ở cấu hình hiện tại.")
        else:
            st.caption(f"Seed {'thủ công' if mode == 'Manual exploration' else 'mạnh nhất chưa dùng'}: "
                       f"(y,x)={seed}, M={data['magnitude'][seed]:.4f}, θlevel={degrees(data['angles'][seed]):.1f}°.")
            st.button("Start manual region" if mode == "Manual exploration" else
                      "Start strongest-seed selection / region", key="lsd_lab_start",
                      on_click=lambda: s.__setitem__("lsd_lab_started", True))
            if region:
                _replay_controls(region, "lsd_lab_replay")
            if region and replay and replay["count"] == len(region["events"]):
                st.success(f"LSR {len(region['pixels'])} pixel đã hoàn thành từ BFS thật. "
                           "Đổi sang workspace B để xem fit/NFA của official strongest-seed scan." if mode != "Manual exploration" else
                           f"LSR thủ công có {len(region['pixels'])} pixel; không dùng NFA toàn ảnh cho vùng này.")
        st.caption(PRESET_NOTES[s.lsd_lab_preset][1] + " " + PRESET_NOTES[s.lsd_lab_preset][2])
    else:
        st.subheader("B · Segment & NFA Explorer")
        if not data["regions"]:
            st.warning("Không có candidate đủ gradient ở tham số này.")
            _chart(data, key="lsd_lab_chart", view="Intensity")
        else:
            options = range(len(data["regions"]))
            st.selectbox("Computed candidate", options, key="lsd_lab_candidate", on_change=_candidate_change,
                         format_func=lambda i: f"#{i+1} · seed {data['regions'][i]['seed']} · "
                         f"n={data['regions'][i]['fit']['n']}, k={data['regions'][i]['fit']['k']}")
            region = data["regions"][s.lsd_lab_candidate]
            if s.get("lsd_presentation", False):
                with st.container(key="lsd_nfa_panels"):
                    left, right = st.columns([1.1, 1], gap="medium")
                    with left:
                        _chart(data, key="lsd_lab_chart", view="Intensity", region=region,
                               fit=True, rectangle=True, nfa=True,
                               show_decision=bool(s.lsd_lab_nfa_decision))
                    with right:
                        _nfa_microscope(data, region, "lsd_lab_nfa")
            else:
                _chart(data, key="lsd_lab_chart", view="Intensity", region=region,
                       fit=True, rectangle=True, nfa=True, show_decision=bool(s.lsd_lab_nfa_decision))
                with st.expander("Hình học fitted segment / rectangle"):
                    _fit_metrics(region)
                _nfa_microscope(data, region, "lsd_lab_nfa")
            if s.lsd_lab_nfa_decision:
                st.metric("N_T · full-image grown LSRs", data["tests"])
                st.caption(f"{sum(r['accept'] for r in data['regions'])} accepted / {data['tests']} tested. "
                           "A/B là #1/#2 trong preset reference; đổi preset/τ có thể thay thứ tự và kết quả.")
                with st.expander("So sánh tất cả candidate từ ảnh hiện tại"):
                    st.dataframe([{"#": i+1, "seed": r["seed"], "n": r["fit"]["n"],
                                   "k": r["fit"]["k"], "NFA": format_probability(r["log_nfa"]),
                                   "decision": "ACCEPT" if r["accept"] else "REJECT"}
                                  for i, r in enumerate(data["regions"])], hide_index=True)
        st.caption(PRESET_NOTES[s.lsd_lab_preset][1] + " " + PRESET_NOTES[s.lsd_lab_preset][2])
    st.button("Reset to reference", key="lsd_lab_reset", on_click=_reset_lab)
