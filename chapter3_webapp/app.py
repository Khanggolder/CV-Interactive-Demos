"""Interactive Chapter 3 tutorials: RANSAC, Hough, Sobel, Canny, Circle and LSD."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from utils.ransac import (
    GeneratedDataset,
    classify_inliers,
    fit_ordinary_least_squares,
    guided_dataset,
    line_equation,
    perpendicular_distances,
    generate_dataset,
    run_ransac,
)
from utils.visualization import (
    LAB_STAGES,
    PLOT_CONFIG,
    lab_stage_figure,
    point_cloud_figure as _point_cloud_figure,
    trial_score_figure,
)
from tutorials.hough import guided_tutorial as hough_guided_tutorial, experiment_lab as hough_experiment_lab
from tutorials.sobel import guided_tutorial as sobel_guided_tutorial, experiment_lab as sobel_experiment_lab
from tutorials.canny import guided_tutorial as canny_guided_tutorial, experiment_lab as canny_experiment_lab
from tutorials.hough_circle import guided_tutorial as circle_guided_tutorial, experiment_lab as circle_experiment_lab
from tutorials.lsd import guided_tutorial as lsd_guided_tutorial, experiment_lab as lsd_experiment_lab


st.set_page_config(
    page_title="Chapter 3 · Interactive Vision Lab",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      :root {
        --cv-navy: #123b5d;
        --cv-blue: #1996c8;
        --cv-orange: #e85d2a;
        --cv-green: #14966f;
        --cv-ink: #172033;
        --cv-muted: #52657a;
        --cv-line: #cbd9e2;
        --cv-surface: #ffffff;
      }
      .block-container {max-width: 1320px; padding-top: 3.75rem; padding-bottom: 4rem;}
      h1, h2, h3 {color: var(--cv-navy) !important; letter-spacing: -.02em;}
      p, li, label, [data-testid="stCaptionContainer"] {color: var(--cv-ink);}
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: #334155 !important; opacity: 1 !important;
      }
      .chapter-kicker {
        color: var(--cv-orange); font-size: .78rem; font-weight: 850;
        letter-spacing: .14em; margin-bottom: .35rem; text-transform: uppercase;
      }
      .hero-copy {color: #334155; font-size: 1.04rem; max-width: 900px;}
      .sidebar-brand {
        color: white; background: linear-gradient(135deg, #123b5d, #0f6b78);
        border-radius: 14px; box-shadow: 0 8px 22px rgba(18,59,93,.18);
        margin: .35rem 0 1.2rem; padding: 1rem 1.1rem;
      }
      .sidebar-brand strong {display:block; font-size:1.15rem; letter-spacing:.05em;}
      .sidebar-brand span {color:#d8eef1; display:block; font-size:.82rem; margin-top:.2rem;}
      [data-testid="stSidebar"] {background:#edf3f6; border-right:1px solid var(--cv-line);}
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color:#26394f !important;}
      [data-testid="stSidebar"] .sidebar-brand span {color:#d8eef1 !important;}
      .stepper {display:grid; grid-template-columns:repeat(10,1fr); gap:.35rem; margin:.7rem 0 1.2rem;}
      .step {
        background:#e5edf2; border:1px solid #cbd9e2; border-radius:8px;
        color:#52657a; font-size:.72rem; font-weight:800; padding:.48rem .2rem;
        text-align:center;
      }
      .step.done {background:#dff3ed; border-color:#9ed8c8; color:#0f6b55;}
      .step.active {background:#fff0e8; border:2px solid var(--cv-orange); color:#a53b14;}
      .hough-stepper {grid-template-columns:repeat(12,1fr);}
      .hough-stepper .step {font-size:.66rem;}
      .sobel-stepper {grid-template-columns:repeat(12,1fr);}
      .sobel-stepper .step {font-size:.65rem;}
      .canny-stepper {grid-template-columns:repeat(14,1fr);}
      .canny-stepper .step {font-size:.61rem; min-width:0; overflow-wrap:anywhere;}
      .circle-stepper {grid-template-columns:repeat(14,1fr);}
      .circle-stepper .step {font-size:.61rem; min-width:0; overflow-wrap:anywhere;}
      .lsd-stepper {grid-template-columns:repeat(14,1fr);}
      .lsd-stepper .step {font-size:.65rem; min-width:0; overflow-wrap:anywhere;}
      .st-key-lsd_guided_chart, .st-key-lsd_lab_chart {max-width:780px; margin-inline:auto;}
      .st-key-circle_image_only {max-width:760px; margin-inline:auto;}
      .st-key-canny_main_chart {max-width:820px;}
      .st-key-canny_stage_patch, .st-key-canny_nms_patch {max-width:500px;}
      .sobel-matrix {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:3px; max-width:360px; margin:.35rem 0 .7rem;}
      .sobel-cell {display:grid; place-items:center; min-height:52px; border:2px solid var(--matrix-color); border-radius:7px; background:#f5fafc; color:#172033; font-size:1.12rem; font-weight:800; font-variant-numeric:tabular-nums;}
      .sobel-cell.active {background:#fff2bd; border-color:#c48500; box-shadow:inset 0 0 0 2px #c48500;}
      .st-key-sobel_image_chart, .st-key-sobel_map_chart {max-width:760px;}
      .st-key-sobel_vector_chart {max-width:480px;}
      @media (min-width:1200px) {
        .st-key-sobel_panels .sobel-cell {min-height:64px; font-size:1.35rem;}
      }
      .mobile-progress {display:none;}
      .progress-track {background:#d8e4eb; border-radius:99px; height:6px; margin-top:.35rem;}
      .progress-track span {display:block; height:100%; border-radius:99px; background:var(--cv-orange);}
      .lab-stage-strip {display:flex; flex-wrap:wrap; gap:.35rem; margin:.3rem 0 .75rem;}
      .lab-stage-strip span {border:1px solid var(--cv-line); border-radius:6px; color:var(--cv-muted); font-size:.8rem; padding:.24rem .5rem;}
      .lab-stage-strip span.active {background:#fff0e8; border-color:var(--cv-orange); color:#a53b14; font-weight:800;}
      .teaching-card {
        background:white; border:1px solid var(--cv-line); border-radius:12px;
        box-shadow:0 3px 12px rgba(18,59,93,.05); margin:.4rem 0 1rem; padding:1rem 1.15rem;
      }
      .teaching-card h3 {font-size:1.12rem; margin:0 0 .35rem;}
      .question {
        background:#fff4e9; border-left:5px solid var(--cv-orange); border-radius:8px;
        color:#713515; font-size:1.02rem; padding:.8rem 1rem;
      }
      .takeaway {
        background:#e9f7f2; border-left:5px solid var(--cv-green); border-radius:8px;
        color:#0d5946; font-size:1rem; padding:.8rem 1rem;
      }
      .formula-box {
        background:#f0f7fa; border:1px solid #bfdce5; border-radius:10px;
        color:var(--cv-navy); font-size:1.08rem; padding:.8rem 1rem; text-align:center;
      }
      [data-testid="stMetric"] {
        background:white; border:1px solid var(--cv-line); border-radius:10px;
        box-shadow:0 2px 8px rgba(18,59,93,.04); min-height:88px; padding:.65rem .75rem;
      }
      [data-testid="stMetricLabel"] p {color:#52657a !important; font-weight:700;}
      [data-testid="stMetricValue"] {color:var(--cv-ink); font-size:1.45rem;}
      .stButton > button {
        background:white; border:1px solid #afc2ce; border-radius:9px;
        color:var(--cv-navy); font-weight:750;
      }
      .stButton > button:hover {background:#eaf5f6; border-color:#0f6b78;}
      div[data-testid="stPlotlyChart"] {background:white; border:1px solid var(--cv-line); border-radius:12px;}
      [data-testid="stDataFrame"] {border:1px solid var(--cv-line); border-radius:10px; overflow:hidden;}
      @media (max-width: 900px) {
        .stepper {display:none;}
        .mobile-progress {display:block; margin:.25rem 0 .65rem; color:var(--cv-navy); font-weight:800;}
        .block-container {padding-top:2.55rem;}
        h1 {font-size:2.2rem !important; line-height:1.12 !important; margin-bottom:.35rem !important;}
        .chapter-kicker {font-size:.69rem; margin-bottom:.15rem;}
        .hero-copy {font-size:.94rem; line-height:1.35;}
        .st-key-guided_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.4rem !important;}
        .st-key-guided_nav [data-testid="stColumn"] {min-width:0 !important; width:33.333% !important; flex:1 1 0 !important;}
        .st-key-guided_nav button {padding:.3rem .25rem; font-size:.85rem; min-height:2.2rem;}
        .st-key-hough_panels [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-hough_panels [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
        .st-key-sobel_panels [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-sobel_panels [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
        .st-key-canny_panels [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-canny_panels [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
        .st-key-circle_panels [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-circle_panels [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
        .st-key-lsd_replay_actions [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.35rem !important;}
        .st-key-lsd_replay_actions [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .2rem) !important; flex:0 0 calc(50% - .2rem) !important;}
        .st-key-lsd_nfa_panels > [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-lsd_nfa_panels > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
      }
      @media (max-width: 600px) {
        h1 {font-size:1.85rem !important;}
        .block-container {padding-left:1rem; padding-right:1rem;}
        h2, h3 {line-height:1.2 !important;}
        .hero-copy {font-size:.86rem; line-height:1.25;}
        .question {padding:.55rem .7rem; font-size:.92rem;}
        .st-key-lab_run [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.4rem !important;}
        .st-key-lab_run [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .2rem) !important; flex:0 0 calc(50% - .2rem) !important;}
        .st-key-lab_run button {font-size:.81rem; padding:.25rem; min-height:2.35rem;}
        .st-key-lab_stage_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.4rem !important;}
        .st-key-lab_stage_nav [data-testid="stColumn"] {min-width:0 !important; width:50% !important; flex:1 1 0 !important;}
        .st-key-lab_stage_nav button {font-size:.85rem; padding:.25rem; min-height:2.25rem;}
        .st-key-hough_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.3rem !important;}
        .st-key-hough_nav [data-testid="stColumn"] {min-width:0 !important; width:33.3% !important; flex:1 1 0 !important;}
        .st-key-hough_nav button {font-size:.78rem; padding:.25rem; min-height:2.2rem;}
        .st-key-hough_vote_actions [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.35rem !important;}
        .st-key-hough_vote_actions [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .2rem) !important; flex:0 0 calc(50% - .2rem) !important;}
        .st-key-hough_vote_actions button {font-size:.78rem; padding:.2rem;}
        .st-key-sobel_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.3rem !important;}
        .st-key-sobel_nav [data-testid="stColumn"] {min-width:0 !important; width:33.3% !important; flex:1 1 0 !important;}
        .st-key-sobel_nav button {font-size:.78rem; padding:.25rem;}
        .st-key-sobel_operations [data-testid="stHorizontalBlock"],
        .st-key-sobel_center_controls [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.3rem !important;}
        .st-key-sobel_operations [data-testid="stColumn"],
        .st-key-sobel_center_controls [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .15rem) !important; flex:0 0 calc(50% - .15rem) !important;}
        .st-key-sobel_operations button, .st-key-sobel_center_controls button {font-size:.78rem; padding:.25rem;}
        .st-key-canny_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.3rem !important;}
        .st-key-canny_nav [data-testid="stColumn"] {min-width:0 !important; width:33.3% !important; flex:1 1 0 !important;}
        .st-key-canny_nav button {font-size:.78rem; padding:.25rem;}
        .st-key-canny_bfs_controls [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.35rem !important;}
        .st-key-canny_bfs_controls [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .18rem) !important; flex:0 0 calc(50% - .18rem) !important;}
        .st-key-canny_bfs_controls button {font-size:.78rem; padding:.25rem; min-height:2.4rem;}
        .st-key-circle_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.3rem !important;}
        .st-key-circle_nav [data-testid="stColumn"] {min-width:0 !important; width:33.3% !important; flex:1 1 0 !important;}
        .st-key-circle_nav button {font-size:.78rem; padding:.25rem;}
        .st-key-circle_vote_actions [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:.3rem !important;}
        .st-key-circle_vote_actions [data-testid="stColumn"] {min-width:0 !important; width:calc(50% - .15rem) !important; flex:0 0 calc(50% - .15rem) !important;}
        .st-key-circle_vote_actions button {font-size:.76rem; padding:.22rem;}
        .st-key-circle_primary_controls [data-testid="stHorizontalBlock"] {flex-direction:column !important;}
        .st-key-circle_primary_controls [data-testid="stColumn"] {width:100% !important; min-width:0 !important;}
        .st-key-lsd_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important; gap:.3rem !important;}
        .st-key-lsd_nav [data-testid="stColumn"] {min-width:0 !important; width:33.3% !important; flex:1 1 0 !important;}
        .st-key-lsd_nav button, .st-key-lsd_replay_actions button {font-size:.77rem; padding:.22rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


PRESETS = {
    "1 · CLEAN DATA": {
        "inliers": 32,
        "outliers": 5,
        "noise": 0.08,
        "epsilon": 0.20,
        "seed": 11,
        "iterations": 25,
        "question": "Ít outlier thì RANSAC cần bao nhiêu lần lấy mẫu để tìm được cấu trúc chính?",
        "action": "Chạy từng iteration và theo dõi best consensus.",
        "takeaway": "Khi tỷ lệ inlier cao, xác suất lấy đúng hai inlier cũng cao.",
    },
    "2 · MANY OUTLIERS": {
        "inliers": 20,
        "outliers": 40,
        "noise": 0.10,
        "epsilon": 0.22,
        "seed": 23,
        "iterations": 80,
        "question": "Khi outlier chiếm đa số, những candidate đầu tiên có đáng tin không?",
        "action": "Chạy một iteration trước, sau đó chạy thêm một batch.",
        "takeaway": "Nhiều outlier làm sample tốt hiếm hơn; N nhỏ có thể bỏ lỡ mô hình.",
    },
    "3 · WRONG EPSILON": {
        "inliers": 28,
        "outliers": 14,
        "noise": 0.13,
        "epsilon": 0.035,
        "seed": 31,
        "iterations": 45,
        "question": "Điều gì xảy ra khi ε nhỏ hơn nhiễu thật hoặc lớn đến mức nhận cả outlier?",
        "action": "So sánh ε = 0.035 với ε = 0.90 trên cùng dữ liệu.",
        "takeaway": "ε cần phản ánh mức nhiễu đo; quá nhỏ mất inlier, quá lớn nhận nhầm outlier.",
    },
}


def page_intro(title: str, purpose: str) -> None:
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("presentation_mode", False):
        st.markdown(f'<div class="hero-copy">{purpose}</div>', unsafe_allow_html=True)


def point_cloud_figure(*args, **kwargs):
    return _point_cloud_figure(
        *args, presentation=bool(st.session_state.get("presentation_mode", False)), **kwargs
    )


def teaching_card(title: str, text: str, class_name: str = "teaching-card") -> None:
    st.markdown(
        f'<div class="{class_name}"><h3>{title}</h3><div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_stepper(active: int) -> None:
    labels = ("Quan sát", "Dự đoán", "Sample", "Khoảng cách", "Phân loại", "Score", "Lặp", "Refit", "So sánh", "Kết luận")
    cells = []
    for index, label in enumerate(labels, start=1):
        state = "active" if index == active else ("done" if index < active else "")
        cells.append(f'<div class="step {state}">{index}<br>{label}</div>')
    st.markdown(f'<div class="stepper">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mobile-progress">Tiến độ học · {active}/10'
        f'<div class="progress-track"><span style="width:{active * 10}%"></span></div></div>',
        unsafe_allow_html=True,
    )


def _guided_previous() -> None:
    st.session_state.guided_step = max(1, st.session_state.guided_step - 1)


def _guided_next() -> None:
    st.session_state.guided_step = min(10, st.session_state.guided_step + 1)


def _guided_reset() -> None:
    st.session_state.guided_step = 1
    st.session_state.pop("guided_prediction", None)
    st.session_state.pop("guided_membership_prediction", None)
    st.session_state.pop("guided_reveal", None)


def _guided_reveal() -> None:
    st.session_state.guided_reveal = True


def guided_navigation() -> int:
    if "guided_step" not in st.session_state:
        st.session_state.guided_step = 1
    with st.container(key="guided_nav"):
        previous, next_column, reset = st.columns(3)
        with previous:
            st.button(
                "← Previous",
                disabled=st.session_state.guided_step == 1,
                width="stretch",
                on_click=_guided_previous,
            )
        with next_column:
            next_disabled = (
                st.session_state.guided_step == 10
                or (st.session_state.guided_step == 2 and not st.session_state.get("guided_prediction"))
                or (st.session_state.guided_step == 5 and not st.session_state.get("guided_reveal", False))
            )
            st.button("Next →", disabled=next_disabled, width="stretch", on_click=_guided_next)
        with reset:
            st.button("Reset", width="stretch", on_click=_guided_reset)
    return int(st.session_state.guided_step)


def guided_tutorial() -> None:
    page_intro(
        "RANSAC · Guided Tutorial",
        "Đi từng bước từ point cloud có outlier đến consensus, TLS refit và so sánh với Ordinary Least Squares.",
    )
    step = guided_navigation()
    render_stepper(step)

    points = guided_dataset().points
    epsilon = 0.22
    result = run_ransac(points, epsilon=epsilon, seed=7647, iterations=3)
    first = result.trials[0]
    best = result.best_trial
    assert best is not None
    ols = fit_ordinary_least_squares(points)

    titles = {
        1: "Quan sát point cloud",
        2: "Dự đoán trước khi sample",
        3: "Hai điểm tạo một candidate line",
        4: "Khoảng cách vuông góc và dải ε",
        5: "Phân loại inlier / outlier",
        6: "Consensus score",
        7: "Lặp lại và giữ model tốt nhất",
        8: "Refit bằng toàn bộ consensus inlier",
        9: "Ordinary Least Squares và RANSAC",
        10: "Kết luận và giới hạn",
    }
    st.subheader(f"Bước {step}/10 · {titles[step]}")

    if step == 1:
        figure = point_cloud_figure(points, title="20 điểm quan sát · có outlier")
        st.plotly_chart(figure, config=PLOT_CONFIG)
        teaching_card(
            "Nhiệm vụ",
            "Tìm một đường thẳng mô tả cấu trúc chiếm ưu thế. Màu điểm chưa tiết lộ nhãn sinh dữ liệu.",
        )

    elif step == 2:
        st.markdown('<div class="question"><b>Dự đoán:</b> Hai điểm chọn ngẫu nhiên có luôn tạo ra model tốt không?</div>', unsafe_allow_html=True)
        st.radio(
            "Chọn dự đoán trước khi sang bước tiếp theo",
            ("Có, hai điểm luôn đủ", "Không, sample có thể chứa outlier", "Chưa chắc"),
            index=None,
            key="guided_prediction",
        )
        if st.session_state.get("guided_prediction"):
            st.info("Đã ghi nhận. Sang bước 3 để kiểm tra bằng phép tính thật.")
        figure = point_cloud_figure(points, title="Bạn chưa biết cặp nào sẽ được chọn")

    elif step == 3:
        teaching_card(
            "Sample đã chọn",
            f"Iteration 1 lấy điểm #{first.pair[0]} và #{first.pair[1]}. Candidate: <code>{line_equation(first.model)}</code>",
        )
        figure = point_cloud_figure(
            points, title="Candidate từ đúng hai điểm", model=first.model, sample_pair=first.pair
        )

    elif step == 4:
        st.markdown(
            '<div class="formula-box">d = |a·x + b·y + c| / √(a² + b²) &nbsp;&nbsp;·&nbsp;&nbsp; inlier khi d ≤ ε</div>',
            unsafe_allow_html=True,
        )
        st.caption("Đường nối chấm cho thấy khoảng cách vuông góc, không phải vertical residual.")
        figure = point_cloud_figure(
            points,
            title=f"Dải consensus ε = {epsilon:.2f}",
            model=first.model,
            epsilon=epsilon,
            sample_pair=first.pair,
            show_distances=True,
        )

    elif step == 5:
        revealed = bool(st.session_state.get("guided_reveal", False))
        if not revealed:
            st.markdown('<div class="question">Trước khi reveal: dải ε sẽ chứa nhiều hay ít điểm?</div>', unsafe_allow_html=True)
        st.radio(
            "Dự đoán số điểm trong dải ε",
            ("Nhiều", "Ít", "Chưa chắc"),
            index=None,
            key="guided_membership_prediction",
        )
        st.button(
            "Reveal phân loại",
            type="primary",
            disabled=not st.session_state.get("guided_membership_prediction"),
            on_click=_guided_reveal,
        )
        if revealed:
            st.success("Xanh: d ≤ ε. Cam: d > ε. Hai điểm sample luôn nằm trên candidate nếu sample hợp lệ.")
        figure = point_cloud_figure(
            points,
            title="Membership được tính từ khoảng cách",
            model=first.model,
            epsilon=epsilon,
            inlier_mask=first.inlier_mask if revealed else None,
            sample_pair=first.pair,
        )

    elif step == 6:
        columns = st.columns(3)
        columns[0].metric("Consensus", f"{first.score} / {len(points)}")
        columns[1].metric("ε", f"{epsilon:.2f}")
        columns[2].metric("Sample", f"#{first.pair[0]}, #{first.pair[1]}")
        teaching_card(
            "Quan sát",
            "Candidate đầu tiên chỉ nhận được ít ủng hộ. Một sample hợp lệ về hình học vẫn có thể tạo model kém.",
            "takeaway",
        )
        figure = point_cloud_figure(
            points,
            title="Candidate score lấy trực tiếp từ d ≤ ε",
            model=first.model,
            epsilon=epsilon,
            inlier_mask=first.inlier_mask,
            sample_pair=first.pair,
        )

    elif step == 7:
        teaching_card(
            "Ba lần lấy mẫu có seed 7647",
            "Mỗi cặp tạo model và consensus mới. App giữ trial có score lớn nhất; residual chỉ dùng để phá hòa.",
        )
        score_column, plot_column = st.columns([0.38, 0.62])
        with score_column:
            st.plotly_chart(
                trial_score_figure(
                    result.trials, best.iteration,
                    presentation=bool(st.session_state.get("presentation_mode", False)),
                ),
                config=PLOT_CONFIG,
            )
        with plot_column:
            figure = point_cloud_figure(
                points,
                title=f"Best: iteration {best.iteration}",
                model=best.model,
                epsilon=epsilon,
                inlier_mask=best.inlier_mask,
                sample_pair=best.pair,
            )

    elif step == 8:
        candidate_error = float(np.sqrt(np.mean(np.square(best.distances[best.inlier_mask]))))
        refit_distances = perpendicular_distances(points[best.inlier_mask], result.refit_model)
        refit_error = float(np.sqrt(np.mean(np.square(refit_distances))))
        metrics = st.columns(3)
        metrics[0].metric("Consensus dùng để refit", best.score)
        metrics[1].metric("Candidate RMS ⟂", f"{candidate_error:.4f}")
        metrics[2].metric(
            "TLS refit RMS ⟂", f"{refit_error:.4f}",
            delta=f"{refit_error - candidate_error:+.4f}",
            delta_color="inverse",
        )
        st.caption(
            "Δ = TLS − candidate: RMS vuông góc giảm là cải thiện (mũi tên xuống, màu xanh). "
            "Hai RMS đo trên cùng winning consensus; "
            "TLS/PCA dùng toàn bộ tập đó để refit, không dùng nhãn sinh dữ liệu."
        )
        figure = point_cloud_figure(
            points,
            title="Candidate và TLS refit",
            model=best.model,
            epsilon=epsilon,
            inlier_mask=best.inlier_mask,
            refit_model=result.refit_model,
        )

    elif step == 9:
        ols_consensus = int(classify_inliers(points, ols, epsilon).sum())
        metrics = st.columns(3)
        metrics[0].metric("RANSAC consensus", best.score)
        metrics[1].metric("OLS points trong ε", ols_consensus)
        metrics[2].metric("Tổng số điểm", len(points))
        teaching_card(
            "Vì sao OLS bị kéo lệch?",
            "OLS y-on-x tối thiểu hóa vertical residual bình phương trên mọi điểm. Outlier xa có ảnh hưởng lớn; RANSAC tìm consensus trước rồi mới refit.",
        )
        figure = point_cloud_figure(
            points,
            title="TLS consensus refit so với OLS trên toàn bộ dữ liệu",
            inlier_mask=best.inlier_mask,
            refit_model=result.refit_model,
            ols_model=ols,
        )

    else:
        st.markdown(
            '<div class="takeaway"><b>Kết luận:</b> RANSAC thành công khi ít nhất một sample tạo model có consensus tốt, rồi TLS/PCA dùng toàn bộ consensus để refit.</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            st.markdown("""
            **Điểm mạnh**

            - Chịu được nhiều outlier.
            - Không rời rạc hóa không gian tham số.
            - Cùng ý tưởng có thể áp dụng cho homography và pose.
            """)
        with right:
            st.markdown("""
            **Giới hạn**

            - Kết quả phụ thuộc seed, ε và số iteration.
            - Một số N hữu hạn không bảo đảm sample tốt.
            - MVP chỉ tìm một đường thẳng; dữ liệu nhiều cấu trúc cần phương pháp mở rộng.
            """)
        st.latex(r"N = \frac{\log(1-p)}{\log(1-(1-e)^s)}")
        st.caption("Công thức ước lượng N cần tỷ lệ outlier e. Trong thực tế e thường chỉ là ước lượng, nên xác suất thành công không phải bảo đảm tuyệt đối.")
        figure = point_cloud_figure(
            points,
            title="Kết quả cuối cùng",
            inlier_mask=best.inlier_mask,
            refit_model=result.refit_model,
            ols_model=ols,
        )

    if step != 1:
        st.plotly_chart(figure, config=PLOT_CONFIG)
    with st.expander("Giải thích số học", expanded=False):
        st.code(
            "a*x + b*y + c = 0\n"
            "d = abs(a*x + b*y + c) / sqrt(a*a + b*b)\n"
            "inlier = (d <= epsilon)\n"
            "best = max(consensus score)\n"
            "final = TLS/PCA(best consensus)",
            language="text",
        )


def _preset_values(name: str, epsilon_override: float | None = None) -> None:
    preset = PRESETS[name]
    st.session_state.lab_inliers = preset["inliers"]
    st.session_state.lab_outliers = preset["outliers"]
    st.session_state.lab_noise = preset["noise"]
    st.session_state.lab_epsilon = epsilon_override if epsilon_override is not None else preset["epsilon"]
    st.session_state.lab_seed = preset["seed"]
    st.session_state.lab_iterations = preset["iterations"]
    st.session_state.lab_completed = 0
    st.session_state.lab_prediction = None
    _make_lab_dataset()


def _make_lab_dataset() -> None:
    generated = generate_dataset(
        int(st.session_state.lab_inliers),
        int(st.session_state.lab_outliers),
        float(st.session_state.lab_noise),
        int(st.session_state.lab_seed),
    )
    st.session_state.lab_points = generated.points
    st.session_state.lab_active_config = {
        "inliers": int(st.session_state.lab_inliers),
        "outliers": int(st.session_state.lab_outliers),
        "noise": float(st.session_state.lab_noise),
        "seed": int(st.session_state.lab_seed),
    }
    st.session_state.lab_completed = 0
    st.session_state.lab_prediction = None
    st.session_state.lab_stage = LAB_STAGES[0]


def initialize_lab() -> None:
    defaults = PRESETS["1 · CLEAN DATA"]
    for key, value in (
        ("lab_inliers", defaults["inliers"]),
        ("lab_outliers", defaults["outliers"]),
        ("lab_noise", defaults["noise"]),
        ("lab_epsilon", defaults["epsilon"]),
        ("lab_seed", defaults["seed"]),
        ("lab_iterations", defaults["iterations"]),
        ("lab_completed", 0),
        ("lab_stage", LAB_STAGES[0]),
    ):
        st.session_state.setdefault(key, value)
    if "lab_points" not in st.session_state:
        _make_lab_dataset()


def _reset_lab_experiment() -> None:
    st.session_state.lab_completed = 0
    st.session_state.lab_prediction = None
    st.session_state.lab_stage = LAB_STAGES[0]


def _advance_lab_stage(direction: int) -> None:
    current = LAB_STAGES.index(st.session_state.lab_stage)
    st.session_state.lab_stage = LAB_STAGES[max(0, min(len(LAB_STAGES) - 1, current + direction))]


def _lab_observation(
    candidate, best, total: int, epsilon: float, noise: float,
    completed: int, generated_inliers: int, generated_outliers: int,
) -> str:
    """Interpret trial state without consulting per-point generated labels."""
    if candidate.iteration == best.iteration:
        outcome = f"Candidate hiện tại được giữ làm best: {candidate.score}/{total} điểm có d ≤ ε."
    else:
        outcome = (
            f"Candidate hiện tại chỉ có {candidate.score}/{total} điểm trong ε; "
            f"best ở iteration {best.iteration} có {best.score}/{total}."
        )
    notes = [outcome]
    if generated_outliers >= generated_inliers:
        pair_probability = generated_inliers * (generated_inliers - 1) / (total * (total - 1))
        notes.append(
            f"Theo cấu hình sinh dữ liệu, cơ hội chọn hai điểm được sinh là inlier chỉ "
            f"{pair_probability:.1%} mỗi trial; đây không phải xác suất thành công bảo đảm."
        )
    if noise > 0 and epsilon < noise * 0.5:
        notes.append("ε nhỏ so với noise σ tạo dữ liệu: dải hẹp có thể bỏ sót điểm thuộc cấu trúc.")
    elif noise > 0 and epsilon > max(0.6, noise * 4):
        notes.append("ε rất rộng: consensus cao có thể nhận cả outlier, không chứng minh line đúng.")
    if completed <= 2:
        notes.append("Mới có ít trial; dù score hiện tại cao, chưa thể kết luận đã tìm được model tốt nhất.")
    notes.append("N hữu hạn chỉ tăng cơ hội tìm model tốt, không bảo đảm thành công.")
    return " ".join(notes)


def render_lab_stage(points: np.ndarray, result, epsilon: float) -> str:
    """Render the five-view lens for the last real RANSAC trial."""
    stage = st.session_state.lab_stage
    st.markdown(
        '<div class="lab-stage-strip">' + ''.join(
            f'<span class="{"active" if item == stage else ""}">{index} · {item}</span>'
            for index, item in enumerate(LAB_STAGES, 1)
        ) + '</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="lab_stage_nav"):
        stage_previous, stage_next = st.columns(2)
        with stage_previous:
            st.button(
                "← Pha trước", disabled=stage == LAB_STAGES[0],
                on_click=_advance_lab_stage, args=(-1,), width="stretch",
            )
        with stage_next:
            st.button(
                "Pha tiếp →", disabled=stage == LAB_STAGES[-1],
                on_click=_advance_lab_stage, args=(1,), width="stretch",
            )
    st.plotly_chart(
        lab_stage_figure(
            points, result, epsilon, stage,
            presentation=bool(st.session_state.get("presentation_mode", False)),
        ),
        config=PLOT_CONFIG,
    )
    return stage


def _clamp_lab_completed() -> None:
    st.session_state.lab_completed = min(
        int(st.session_state.get("lab_completed", 0)),
        int(st.session_state.lab_iterations),
    )


def _comparison_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = [
        ("Clean data", PRESETS["1 · CLEAN DATA"], None),
        ("Many outliers", PRESETS["2 · MANY OUTLIERS"], None),
        ("Wrong ε · quá nhỏ", PRESETS["3 · WRONG EPSILON"], 0.035),
        ("Wrong ε · quá lớn", PRESETS["3 · WRONG EPSILON"], 0.90),
    ]
    for label, config, override in variants:
        data = generate_dataset(config["inliers"], config["outliers"], config["noise"], config["seed"])
        epsilon = float(config["epsilon"] if override is None else override)
        result = run_ransac(data.points, epsilon, config["seed"] + 10_000, config["iterations"])
        best_score = result.best_trial.score if result.best_trial else 0
        rows.append(
            {
                "Thí nghiệm": label,
                "Tổng điểm": len(data.points),
                "ε": epsilon,
                "N": config["iterations"],
                "Best consensus": best_score,
                "Tỷ lệ consensus": f"{best_score / len(data.points):.1%}",
            }
        )
    return rows


def experiment_lab() -> None:
    initialize_lab()
    page_intro(
        "RANSAC · Experiment Lab",
        "Điều khiển dữ liệu và thuật toán thật. Mỗi lần chạy dùng cùng seed sẽ tạo cùng point cloud và cùng chuỗi sample.",
    )
    visual_slot = st.container()

    st.subheader("1 · Chọn tình huống học")
    preset_name = st.selectbox("Preset", tuple(PRESETS), key="lab_preset")
    preset = PRESETS[preset_name]
    st.markdown(f'<div class="question"><b>Câu hỏi trước khi chạy:</b> {preset["question"]}</div>', unsafe_allow_html=True)
    preset_columns = st.columns(2)
    with preset_columns[0]:
        st.button("Nạp preset", on_click=_preset_values, args=(preset_name,), width="stretch")
    if preset_name == "3 · WRONG EPSILON":
        with preset_columns[1]:
            st.button("Nạp ε quá lớn", on_click=_preset_values, args=(preset_name, 0.90), width="stretch")

    with st.expander("2 · Tạo dữ liệu và chỉnh tham số", expanded=False):
        first, second, third = st.columns(3)
        with first:
            st.number_input("Số generated inlier", 2, 100, step=1, key="lab_inliers")
            st.number_input("Số outlier", 0, 100, step=1, key="lab_outliers")
        with second:
            st.slider("Noise σ", 0.0, 0.50, step=0.01, key="lab_noise")
            st.number_input("Random seed", 0, 99999, step=1, key="lab_seed")
        with third:
            st.slider("Epsilon ε", 0.0, 1.20, step=0.005, key="lab_epsilon")
            st.slider(
                "Số iteration tối đa",
                1,
                200,
                step=1,
                key="lab_iterations",
                on_change=_clamp_lab_completed,
            )
        if st.button("Generate dataset", type="primary", on_click=_make_lab_dataset):
            st.success("Đã tạo point cloud mới và reset lịch sử sampling.")
        st.caption("Nhãn sinh dữ liệu chỉ dùng để kiểm tra; không tham gia sampling, scoring hoặc fitting.")

    active = st.session_state.lab_active_config
    st.caption(
        f"Đang dùng: {active['inliers']} generated inlier + {active['outliers']} outlier · "
        f"σ={active['noise']:.2f} · seed={active['seed']}"
    )

    st.subheader("3 · Dự đoán rồi chạy RANSAC")
    st.radio(
        "Candidate tiếp theo có thể đạt consensus nào?",
        ("Thấp", "Trung bình", "Cao"),
        index=None,
        horizontal=True,
        key="lab_prediction",
    )
    with st.expander("Chỉnh batch size", expanded=False):
        batch_size = st.slider("Số iteration trong một batch", 2, 30, 10, key="lab_batch")
    disabled = st.session_state.get("lab_prediction") is None
    with st.container(key="lab_run"):
        run_one, run_batch, run_all, reset = st.columns(4)
        with run_one:
            if st.button("Chạy 1 iteration", disabled=disabled, width="stretch"):
                st.session_state.lab_completed = min(
                    st.session_state.lab_completed + 1, st.session_state.lab_iterations
                )
                st.session_state.lab_stage = LAB_STAGES[0]
        with run_batch:
            if st.button(f"Chạy thêm {batch_size}", disabled=disabled, width="stretch"):
                st.session_state.lab_completed = min(
                    st.session_state.lab_completed + batch_size, st.session_state.lab_iterations
                )
                st.session_state.lab_stage = LAB_STAGES[0]
        with run_all:
            if st.button("Chạy toàn bộ N", disabled=disabled, width="stretch"):
                st.session_state.lab_completed = st.session_state.lab_iterations
                st.session_state.lab_stage = LAB_STAGES[0]
        with reset:
            st.button(
                "Reset experiment",
                width="stretch",
                on_click=_reset_lab_experiment,
            )

    points = np.asarray(st.session_state.lab_points)
    completed = int(st.session_state.lab_completed)
    epsilon = float(st.session_state.lab_epsilon)
    if completed == 0:
        with visual_slot:
            st.caption("Quan sát point cloud, rồi chọn dự đoán và chạy một iteration bên dưới.")
            figure = point_cloud_figure(points, title="Point cloud hiện tại · chưa chạy RANSAC")
            st.plotly_chart(figure, config=PLOT_CONFIG)
    else:
        result = run_ransac(
            points,
            epsilon=epsilon,
            seed=int(active["seed"]) + 10_000,
            iterations=completed,
        )
        candidate = result.trials[-1]
        best = result.best_trial
        assert best is not None
        ols = fit_ordinary_least_squares(points)
        ols_count = int(classify_inliers(points, ols, epsilon).sum())

        stage = render_lab_stage(points, result, epsilon)
        metrics = st.columns(4)
        metrics[0].metric("Đã chạy", f"{completed}/{st.session_state.lab_iterations}")
        metrics[1].metric("Candidate", f"{candidate.score}/{len(points)}")
        metrics[2].metric("Best consensus", f"{best.score}/{len(points)}")
        metrics[3].metric("ε", f"{epsilon:.3f}")
        if stage == "Sample":
            st.caption(
                f"Sample #{candidate.pair[0]} và #{candidate.pair[1]} được chọn bởi seed "
                f"{active['seed'] + 10_000}; chưa vẽ line hay phân loại điểm."
            )
        elif stage == "Fit":
            st.caption(f"Hai điểm sample xác định candidate line: {line_equation(candidate.model)}")
        elif stage == "Measure":
            if len(points) == 2:
                st.caption("Chỉ có hai sample: cả hai nằm trên candidate, nên d≈0; không còn điểm khác để đo.")
            else:
                st.caption(
                    f"Khoảng cách vuông góc d = |a·x+b·y+c|/√(a²+b²); "
                    f"đường chấm minh họa điểm gần ranh giới ε={epsilon:.3f} trên mỗi phía."
                )
        elif stage == "Consensus":
            st.caption(
                f"{candidate.score}/{len(points)} điểm có d ≤ ε; màu xanh là inlier của "
                "candidate hiện tại, màu cam là outlier."
            )
        else:
            st.caption(
                "Vàng đứt: candidate hiện tại · xanh lá: best consensus model · "
                "tím: final TLS refit trên winning consensus. "
                + ("Candidate trùng best ở iteration này." if candidate.iteration == best.iteration else "Candidate hiện tại không thay thế best.")
            )
            st.plotly_chart(
                trial_score_figure(
                    result.trials, best.iteration,
                    presentation=bool(st.session_state.get("presentation_mode", False)),
                ),
                config=PLOT_CONFIG,
            )

        teaching_card(
            "Vì sao candidate này thành công hoặc thất bại?",
            _lab_observation(
                candidate, best, len(points), epsilon, float(active["noise"]), completed,
                int(active["inliers"]), int(active["outliers"]),
            ),
        )

        if result.refit_model is not None:
            best_rms = math.sqrt(best.inlier_error / best.score)
            refit_values = perpendicular_distances(points[best.inlier_mask], result.refit_model)
            refit_rms = float(np.sqrt(np.mean(np.square(refit_values))))
            st.markdown(
                f'<div class="takeaway"><b>Final TLS refit:</b> best trial #{best.iteration} có {best.score} inlier. '
                f'Trên cùng winning consensus, RMS vuông góc giảm từ {best_rms:.4f} xuống {refit_rms:.4f} '
                f'(cải thiện {best_rms - refit_rms:.4f}).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Consensus hiện tại chưa đủ hoặc bị suy biến nên chưa thể TLS refit.")
        st.caption(f"OLS trên mọi điểm: {ols_count}/{len(points)} điểm trong ε. Đây là mô hình so sánh, không tham gia chọn best.")

        with st.expander("Inspect inlier membership", expanded=False):
            final_mask = (
                result.refit_mask
                if result.refit_mask is not None
                else np.zeros(len(points), dtype=bool)
            )
            rows = [
                {
                    "point": index,
                    "x": round(float(point[0]), 4),
                    "y": round(float(point[1]), 4),
                    "d(candidate)": round(float(candidate.distances[index]), 4),
                    "candidate": "inlier" if candidate.inlier_mask[index] else "outlier",
                    "final refit": "inlier" if final_mask[index] else "outlier",
                }
                for index, point in enumerate(points)
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
            st.code(f"Candidate: {line_equation(candidate.model)}\nBest TLS: {line_equation(result.refit_model) if result.refit_model is not None else 'none'}\nOLS: {line_equation(ols)}")

    teaching_card("Điều cần quan sát", str(preset["action"]))
    teaching_card("Kết luận chính", str(preset["takeaway"]), "takeaway")

    st.subheader("4 · So sánh thí nghiệm")
    st.caption("Bảng chỉ xuất hiện khi bạn chủ động reveal; tất cả giá trị được chạy lại từ preset và seed tương ứng.")
    if st.checkbox("Reveal bảng so sánh preset", key="lab_compare_reveal"):
        st.dataframe(_comparison_rows(), width="stretch", hide_index=True)
        st.caption("Consensus lớn không tự động có nghĩa model đúng khi ε quá lớn: dải chấp nhận có thể nuốt cả outlier.")


st.sidebar.markdown("### CHAPTER 3")
chapter_module = st.sidebar.radio("Module", ("RANSAC", "Hough Transform", "Sobel Convolution", "Canny Edge Detector", "Hough Circle Transform", "LSD Line Segment Detector"), key="chapter3_module")
presentation = st.sidebar.toggle(
    "Presentation Mode",
    key={"RANSAC": "presentation_mode", "Hough Transform": "hough_presentation",
         "Sobel Convolution": "sobel_presentation", "Canny Edge Detector": "canny_presentation",
         "Hough Circle Transform": "circle_presentation", "LSD Line Segment Detector": "lsd_presentation"}[chapter_module],
)
if presentation:
    st.markdown(
        """
        <style>
          .block-container {max-width:1600px;}
          [data-testid="stMetricValue"] {font-size:1.9rem;}
          .teaching-card, .takeaway {font-size:1.1rem;}
          @media (max-width:900px) {.block-container {max-width:100%;}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Chế độ trình chiếu · hình và chữ lớn hơn. Tắt tại đây để về bố cục thường.")
    if chapter_module == "Sobel Convolution":
        st.markdown("<style>.st-key-sobel_image_chart, .st-key-sobel_map_chart {max-width:1100px;} .st-key-sobel_vector_chart {max-width:620px;} .sobel-cell {min-height:74px; font-size:1.55rem;}</style>",
                    unsafe_allow_html=True)
    if chapter_module == "Canny Edge Detector":
        st.markdown("<style>.st-key-canny_main_chart {max-width:1100px;} .st-key-canny_stage_patch, .st-key-canny_nms_patch {max-width:650px;}</style>", unsafe_allow_html=True)
    if chapter_module == "Hough Circle Transform":
        st.markdown("<style>.st-key-circle_image_only {max-width:1050px;}</style>", unsafe_allow_html=True)
    if chapter_module == "LSD Line Segment Detector":
        st.markdown("<style>.st-key-lsd_guided_chart, .st-key-lsd_lab_chart {max-width:1100px;}</style>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <strong>CV INTERACTIVE LAB</strong>
          <span>Chapter 3 · Vision Geometry</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.sidebar.header("CHẾ ĐỘ HỌC")
mode = st.sidebar.radio(
    "Mode",
    ("A · Guided Tutorial", "B · Experiment Lab"),
    key={"RANSAC": "app_mode", "Hough Transform": "hough_mode",
         "Sobel Convolution": "sobel_mode", "Canny Edge Detector": "canny_mode",
         "Hough Circle Transform": "circle_mode", "LSD Line Segment Detector": "lsd_mode"}[chapter_module],
    label_visibility="collapsed",
)
if not presentation and chapter_module == "RANSAC":
    st.sidebar.divider()
    st.sidebar.markdown("**Mô hình đường thẳng**")
    st.sidebar.latex(r"a x + b y + c = 0")
    st.sidebar.markdown("**Quy tắc consensus**")
    st.sidebar.latex(r"d_\perp \leq \varepsilon")
    st.sidebar.caption("Mọi kết quả trong app được tính khi tương tác. Không dùng ảnh chụp kết quả cố định.")

if chapter_module == "Hough Transform":
    if mode == "A · Guided Tutorial":
        hough_guided_tutorial()
    else:
        hough_experiment_lab()
elif chapter_module == "Sobel Convolution":
    if mode == "A · Guided Tutorial":
        sobel_guided_tutorial()
    else:
        sobel_experiment_lab()
elif chapter_module == "Canny Edge Detector":
    if mode == "A · Guided Tutorial":
        canny_guided_tutorial()
    else:
        canny_experiment_lab()
elif chapter_module == "Hough Circle Transform":
    if mode == "A · Guided Tutorial":
        circle_guided_tutorial()
    else:
        circle_experiment_lab()
elif chapter_module == "LSD Line Segment Detector":
    if mode == "A · Guided Tutorial":
        lsd_guided_tutorial()
    else:
        lsd_experiment_lab()
elif mode == "A · Guided Tutorial":
    guided_tutorial()
else:
    experiment_lab()
