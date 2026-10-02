"""Projector-friendly interactive demos for Chapter 1 of Image Processing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import UnidentifiedImageError

from utils.fourier import (
    apply_frequency_mask,
    compute_fft,
    create_gaussian_lowpass_mask,
    create_highpass_mask,
    create_lowpass_mask,
    linear_magnitude_spectrum,
    magnitude_spectrum,
    reconstruct_image,
    reconstruction_for_display,
)
from utils.geometry import (
    bilinear_sample_details,
    build_transform_matrix,
    centered_rotation_scale_matrix,
    forward_mapping_occupancy,
    inverse_source_coordinate,
    nearest_sample,
    warp_inverse,
)
from utils.image_ops import (
    adjust_brightness_contrast,
    apply_clahe,
    equalize_histogram,
    gamma_correct,
    load_rgb_image,
    mark_pixel_and_patch,
    to_gray,
)
from utils.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    gaussian_filter,
    median_filter,
)
from utils.visualization import (
    PLOT_CONFIG,
    gamma_curve_figure,
    histogram_figure,
    intensity_profile_figure,
)


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "assets" / "samples"
SLIDING_WINDOW_GIF = SAMPLE_DIR / "SlidingWindow.gif"
SAMPLES: dict[str, tuple[str, str]] = {
    "Ảnh tối": ("low_light.png", "Quan sát histogram lệch trái và thử gamma < 1."),
    "Ảnh sáng": ("bright.png", "Quan sát histogram lệch phải và vùng bị clip sáng."),
    "Tương phản thấp": (
        "low_contrast.png",
        "Histogram hẹp; phù hợp Equalization và CLAHE.",
    ),
    "Tương phản cao": (
        "high_contrast.png",
        "Histogram trải rộng, nhiều vùng sáng–tối tách biệt.",
    ),
    "Nhiều texture / tần số": (
        "texture.png",
        "Phổ Fourier giàu thành phần cao tần.",
    ),
    "Cạnh rõ": (
        "clear_edges.png",
        "Quan sát high-pass, Sobel và kernel phát hiện cạnh.",
    ),
    "Ringing test pattern": (
        "ringing_pattern.png",
        "Ảnh tổng hợp dùng để quan sát Gibbs/ringing quanh cạnh sắc.",
    ),
    "Grid / interpolation pattern": (
        "interpolation_grid.png",
        "Ảnh tổng hợp dùng để quan sát biến đổi hình học và nội suy.",
    ),
}

st.set_page_config(
    page_title="Chapter 1 · Interactive CV Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --cv-ink: #172033;
        --cv-muted: #111827;
        --cv-navy: #123b5d;
        --cv-teal: #0f6b78;
        --cv-orange: #e85d2a;
        --cv-line: #d7e1e8;
        --cv-surface: #ffffff;
        --cv-canvas: #f5f8fa;
      }

      html, body, [class*="css"] {font-size: 17px; color: var(--cv-ink);}
      .stApp {background: var(--cv-canvas);}
      [data-testid="stHeader"] {background: rgba(245, 248, 250, .92);}
      .block-container {padding-top: 3.25rem; padding-bottom: 3rem; max-width: 1500px;}

      h1 {
        font-size: 2.35rem !important;
        line-height: 1.12 !important;
        color: var(--cv-navy) !important;
        letter-spacing: -.025em;
        margin: .2rem 0 .35rem !important;
      }
      h2 {font-size: 1.55rem !important; color: var(--cv-teal) !important;}
      h3 {
        font-size: 1.05rem !important;
        letter-spacing: .055em;
        text-transform: uppercase;
        color: var(--cv-navy) !important;
        border-bottom: 2px solid var(--cv-line);
        padding-bottom: .45rem;
        margin-top: 1.7rem !important;
      }
      p, label, [data-testid="stCaptionContainer"] {color: var(--cv-muted);}
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] * {
        color: #111827 !important;
        opacity: 1 !important;
      }
      figcaption, figcaption * {color: #111827 !important; opacity: 1 !important;}

      .chapter-kicker {
        display: inline-flex;
        position: relative;
        z-index: 2;
        align-items: center;
        border-radius: 999px;
        background: #e3f2f4;
        color: var(--cv-teal);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .11em;
        padding: .35rem .65rem;
      }
      .pixel-section {
        color: var(--cv-navy);
        border-bottom: 2px solid var(--cv-line);
        font-size: .93rem;
        font-weight: 800;
        letter-spacing: .035em;
        margin: .7rem 0 .55rem;
        padding-bottom: .32rem;
        text-transform: uppercase;
      }
      .pixel-stats {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .4rem;
      }
      .pixel-stat {
        background: var(--cv-surface);
        border: 1px solid var(--cv-line);
        border-radius: 8px;
        min-width: 0;
        padding: .52rem .45rem;
        text-align: center;
      }
      .pixel-stat-label {
        color: #334155;
        font-size: .66rem;
        font-weight: 750;
        line-height: 1.15;
        min-height: 1.55rem;
      }
      .pixel-stat-value {
        color: #111827;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
      }
      .pixel-note {
        background: #fff4e9;
        border-left: 4px solid var(--cv-orange);
        border-radius: 6px;
        color: #713515;
        font-size: .84rem;
        margin-top: .5rem;
        padding: .45rem .65rem;
      }
      .pixel-value {
        align-items: baseline;
        background: #eef7f8;
        border: 1px solid #bfdce0;
        border-radius: 8px;
        color: var(--cv-ink);
        display: flex;
        gap: .55rem;
        justify-content: space-between;
        margin-top: .15rem;
        padding: .5rem .7rem;
      }
      .pixel-value-label {font-size: .76rem; font-weight: 750;}
      .pixel-value-number {font-size: 1.08rem; font-weight: 800; white-space: nowrap;}
      .sidebar-brand {
        color: #ffffff;
        background: linear-gradient(135deg, #123b5d 0%, #0f6b78 100%);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: .35rem 0 1.25rem;
        box-shadow: 0 8px 22px rgba(18, 59, 93, .18);
      }
      .sidebar-brand strong {display: block; font-size: 1.18rem; letter-spacing: .04em;}
      .sidebar-brand span {display: block; margin-top: .15rem; color: #d8eef1; font-size: .82rem;}

      [data-testid="stMetric"] {
        background: var(--cv-surface);
        border: 1px solid var(--cv-line);
        border-radius: 10px;
        padding: .7rem .8rem;
        min-height: 88px;
        box-shadow: 0 2px 8px rgba(18, 59, 93, .04);
      }
      [data-testid="stMetricLabel"] p {color: var(--cv-muted) !important; font-weight: 650;}
      [data-testid="stMetricValue"] {font-size: 1.52rem; color: var(--cv-ink);}

      .flow-card {
        background: #eef7f8;
        border: 1px solid #bfdce0;
        border-radius: 12px;
        color: var(--cv-ink);
        padding: 1rem;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
      }
      .flow-label {font-weight: 800; color: var(--cv-teal); letter-spacing: .07em; font-size: .82rem;}
      .formula {font-size: 1.32rem; font-weight: 750; color: #b94319; margin: .8rem 0;}
      .note {
        background: #fff4e9;
        border: 1px solid #fed7aa;
        border-left: 5px solid var(--cv-orange);
        color: #713515;
        padding: .8rem 1rem;
        border-radius: 8px;
        font-size: 1.02rem;
        line-height: 1.55;
      }
      .note b {color: #713515;}
      .arrow {text-align: center; font-size: 1.8rem; color: var(--cv-teal); font-weight: 800;}

      [data-testid="stSidebar"] {
        min-width: 315px;
        max-width: 315px;
        background: #edf3f6;
        border-right: 1px solid var(--cv-line);
      }
      [data-testid="stSidebar"] h2 {font-size: 1rem !important; letter-spacing: .065em; color: var(--cv-navy) !important;}
      [data-testid="stSidebar"] hr {border-color: #c9d6de;}
      [data-testid="stSidebar"] [role="radiogroup"] label {padding: .12rem 0;}
      [data-testid="stSidebar"] [data-testid="stRadio"] label p,
      [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
      [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
        color: #31455e !important;
      }
      [data-testid="stSidebar"] [role="radiogroup"] label,
      [data-testid="stSidebar"] [role="radiogroup"] label span,
      [data-testid="stSidebar"] [role="radiogroup"] label div {
        color: #31455e !important;
      }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #111827 !important;
        opacity: 1 !important;
      }

      div[data-testid="stImage"] {
        background: var(--cv-surface);
        border: 1px solid var(--cv-line);
        border-radius: 11px;
        padding: .4rem;
        box-shadow: 0 3px 12px rgba(18, 59, 93, .06);
      }
      div[data-testid="stImage"] img {border-radius: 7px;}
      [data-testid="stAlert"] {border-radius: 10px; border: 1px solid #c9dce4;}
      [data-testid="stDataFrame"] {border: 1px solid var(--cv-line); border-radius: 10px; overflow: hidden;}
      .katex, .katex * {color: var(--cv-ink) !important;}
      .stButton > button {
        border-radius: 9px;
        border: 1px solid #b8c8d4 !important;
        background: #ffffff !important;
        color: var(--cv-navy) !important;
        font-weight: 700;
      }
      .stButton > button p {color: var(--cv-navy) !important;}
      .stButton > button:hover {
        border-color: var(--cv-teal) !important;
        background: #eaf5f6 !important;
      }
      .stLinkButton > a {
        border-radius: 9px;
        border: 1px solid #c9481c !important;
        background: var(--cv-orange) !important;
        color: #ffffff !important;
        font-weight: 700;
      }
      .stLinkButton > a p, .stLinkButton > a span {color: #ffffff !important;}
      .stLinkButton > a:hover {background: #cc4b20 !important;}

      @media (max-width: 900px) {
        .block-container {padding-top: 2.75rem;}
        h1 {font-size: 1.9rem !important;}
        [data-testid="stSidebar"] {min-width: 285px; max-width: 285px;}
        .pixel-stats {grid-template-columns: repeat(2, minmax(0, 1fr));}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <strong>CV INTERACTIVE LAB</strong>
      <span>Chapter 1 · Image Processing Foundations</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def image_source() -> tuple[np.ndarray, str]:
    """Render the shared image picker and return the current image and label."""
    st.sidebar.header("IMAGE SOURCE")
    source_type = st.sidebar.radio(
        "Nguồn ảnh", ("Ảnh mẫu", "Upload ảnh"), horizontal=True, key="source_type"
    )
    if source_type == "Upload ảnh":
        upload = st.sidebar.file_uploader(
            "PNG, JPG hoặc JPEG", type=("png", "jpg", "jpeg")
        )
        if upload is not None:
            try:
                return load_rgb_image(upload), upload.name
            except (UnidentifiedImageError, OSError):
                st.sidebar.error("Không đọc được file ảnh này.")
        st.sidebar.info("Chưa có file — đang dùng ảnh mẫu.")

    selected = st.sidebar.selectbox("Chọn ảnh mẫu", tuple(SAMPLES), key="sample_name")
    filename, teaching_purpose = SAMPLES[selected]
    st.sidebar.caption(f"Mục đích demo: {teaching_purpose}")
    sample_path = SAMPLE_DIR / filename
    if not sample_path.exists():
        st.error("Thiếu ảnh mẫu. Chạy `python scripts/generate_samples.py`.")
        st.stop()
    return load_rgb_image(sample_path), selected


def page_intro(title: str, purpose: str) -> None:
    st.markdown('<div class="chapter-kicker">CHAPTER 1 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(purpose)


def section_label(text: str) -> None:
    st.markdown(f"### {text}")


def pixel_section_label(text: str) -> None:
    st.markdown(f'<div class="pixel-section">{text}</div>', unsafe_allow_html=True)


def show_image(image: np.ndarray, caption: str) -> None:
    st.image(image, caption=caption, width="stretch", clamp=True)


def reset_bc() -> None:
    st.session_state.pop("bc_alpha", None)
    st.session_state.pop("bc_beta", None)


def set_gamma(value: float) -> None:
    st.session_state.gamma_value = value


def hide_fourier_result() -> None:
    st.session_state.fourier_reveal = False


def hide_noise_result() -> None:
    st.session_state.noise_reveal = False


def hide_geometry_result() -> None:
    st.session_state.geometry_reveal = False


def generate_new_noise() -> None:
    st.session_state.noise_seed = st.session_state.get("noise_seed", 2025) + 1
    hide_noise_result()


def reset_reveal_for_new_image(prefix: str, image: np.ndarray) -> None:
    """Hide one page's output only when its input pixels actually change."""
    fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
    fingerprint_key = f"{prefix}_input_fingerprint"
    reveal_key = f"{prefix}_reveal"
    previous = st.session_state.get(fingerprint_key)
    if previous is not None and previous != fingerprint:
        st.session_state[reveal_key] = False
    st.session_state[fingerprint_key] = fingerprint


def reset_fourier_reveal_for_new_image(image: np.ndarray) -> None:
    """Hide Fourier output only when the input pixels actually change."""
    reset_reveal_for_new_image("fourier", image)


def pixel_explorer(image_rgb: np.ndarray) -> None:
    page_intro("1 · Pixel Explorer", "Ảnh số là một ma trận các giá trị pixel.")
    mode = st.radio("Biểu diễn", ("RGB", "Grayscale"), horizontal=True, key="pixel_mode")
    image = image_rgb if mode == "RGB" else to_gray(image_rgb)
    height, width = image.shape[:2]
    channels = 3 if image.ndim == 3 else 1

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        pixel_section_label("1 · INPUT — ORIGINAL IMAGE")
        image_slot = st.empty()
    with right:
        pixel_section_label("2 · PIXEL DATA")
        stats = (
            ("Kích thước", f"{width}×{height}"),
            ("Channels", str(channels)),
            ("dtype", str(image.dtype)),
            ("Min → Max", f"{image.min()}→{image.max()}"),
            ("Mean", f"{float(image.mean()):.1f}"),
        )
        stats_html = "".join(
            f'<div class="pixel-stat"><div class="pixel-stat-label">{label}</div>'
            f'<div class="pixel-stat-value">{value}</div></div>'
            for label, value in stats
        )
        st.markdown(
            f'<div class="pixel-stats">{stats_html}</div>'
            '<div class="pixel-note">Mỗi vị trí <b>(x, y)</b> trỏ đến một ô trong ma trận ảnh.</div>',
            unsafe_allow_html=True,
        )

        pixel_section_label("3 · INSPECT PIXEL")
        sx, sy = st.columns(2, gap="medium")
        x = sx.slider("x (cột)", 0, width - 1, width // 2)
        y = sy.slider("y (hàng)", 0, height - 1, height // 2)
        value = image[y, x]
        if image.ndim == 3:
            value_label = f"Pixel({x}, {y}) = [R, G, B]"
            value_text = f"[{int(value[0])}, {int(value[1])}, {int(value[2])}]"
        else:
            value_label = f"Pixel({x}, {y}) = I"
            value_text = str(int(value))
        st.markdown(
            f'<div class="pixel-value"><span class="pixel-value-label">{value_label}</span>'
            f'<span class="pixel-value-number">{value_text}</span></div>',
            unsafe_allow_html=True,
        )

        radius = 3
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        patch = image[y0:y1, x0:x1]

        patch_image, patch_values = st.columns([0.72, 1.28], gap="medium")
        with patch_image:
            pixel_section_label("4 · PATCH 7×7")
            st.image(
                patch,
                caption=f"{patch.shape[1]} × {patch.shape[0]}",
                width="stretch",
                clamp=True,
            )
        with patch_values:
            pixel_section_label("5 · MA TRẬN INTENSITY")
            gray_patch = patch if patch.ndim == 2 else to_gray(patch)
            st.dataframe(
                gray_patch,
                width="stretch",
                hide_index=True,
                height="auto",
                row_height=32,
            )
            st.caption("RGB → intensity để bảng 7×7 dễ đọc.")

    marked_image = mark_pixel_and_patch(image, x, y, radius)
    image_slot.image(
        marked_image,
        caption=f"{mode} · dấu đỏ: pixel ({x}, {y}) · khung vàng: patch",
        width="stretch",
        clamp=True,
    )


def brightness_contrast(image_rgb: np.ndarray) -> None:
    page_intro(
        "2 · Brightness & Contrast",
        "Toán tử điểm tuyến tính: thay đổi từng pixel độc lập.",
    )
    mode = st.radio(
        "Không gian xử lý", ("RGB trực tiếp", "Grayscale intensity"), horizontal=True, key="bc_mode"
    )
    image = image_rgb if mode == "RGB trực tiếp" else to_gray(image_rgb)

    section_label("1 · PARAMETER")
    controls = st.columns([2, 2, 1])
    alpha = controls[0].slider(
        "α · contrast", 0.1, 3.0, 1.0, 0.05, key="bc_alpha"
    )
    beta = controls[1].slider(
        "β · brightness", -100, 100, 0, 1, key="bc_beta"
    )
    controls[2].button("↺ Reset Parameters", on_click=reset_bc, width="stretch")
    result = adjust_brightness_contrast(image, alpha, beta)

    section_label("2 · INPUT → TRANSFORMATION → OUTPUT")
    before, transform, after = st.columns([1.1, 0.78, 1.1], gap="large")
    with before:
        show_image(image, "ORIGINAL")
    with transform:
        st.markdown(
            f"""
            <div class="flow-card">
              <div class="flow-label">TRANSFORMATION</div>
              <div class="formula">I<sub>out</sub> = {alpha:.2f} × I<sub>in</sub> {beta:+d}</div>
              <div>clip về <b>[0, 255]</b></div>
            </div>
            <div class="arrow">→</div>
            """,
            unsafe_allow_html=True,
        )
    with after:
        show_image(result, "RESULT")

    section_label("3 · INTERMEDIATE STEP — ONE PIXEL")
    example_control, example_math = st.columns([1, 2], gap="large")
    sample_intensity = example_control.slider(
        "Pixel input I_in", 0, 255, 100, key="bc_sample_intensity"
    )
    raw_value = alpha * sample_intensity + beta
    clipped_value = float(np.clip(raw_value, 0, 255))
    quantized_value = int(clipped_value)
    example_math.markdown(
        f"""
        <div class="flow-card" style="min-height: 90px">
          <div class="flow-label">TÍNH TRÊN MỘT PIXEL</div>
          <div class="formula">I<sub>out</sub> = {alpha:.2f} × {sample_intensity} {beta:+d}
          = {raw_value:.1f} → clip = {clipped_value:.1f} → uint8 = {quantized_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_label("4 · INTERMEDIATE EVIDENCE — HISTOGRAM")
    before_gray = image if image.ndim == 2 else to_gray(image)
    after_gray = result if result.ndim == 2 else to_gray(result)
    before_mean = float(before_gray.mean())
    after_mean = float(after_gray.mean())
    before_std = float(before_gray.std())
    after_std = float(after_gray.std())
    clipped_percent = 100.0 * float(
        np.count_nonzero((result == 0) | (result == 255))
    ) / float(result.size)

    h1, h2 = st.columns(2, gap="large")
    h1.plotly_chart(
        histogram_figure(
            image,
            "Grayscale",
            f"Before · μ={before_mean:.1f}, σ={before_std:.1f}",
            line_color="#123b5d",
        ),
        config=PLOT_CONFIG,
        theme=None,
    )
    h2.plotly_chart(
        histogram_figure(
            result,
            "Grayscale",
            f"After · μ={after_mean:.1f}, σ={after_std:.1f}",
            line_color="#e85d2a",
        ),
        config=PLOT_CONFIG,
        theme=None,
    )
    evidence_mean, evidence_std, evidence_clip = st.columns(3)
    evidence_mean.metric(
        "Mean intensity · μ",
        f"{before_mean:.1f} → {after_mean:.1f}",
        f"{after_mean - before_mean:+.1f}",
    )
    evidence_std.metric(
        "Độ phân tán · σ",
        f"{before_std:.1f} → {after_std:.1f}",
        f"{after_std - before_std:+.1f}",
    )
    evidence_clip.metric("Output tại 0 / 255", f"{clipped_percent:.1f}%")

    messages = []
    messages.append(
        "α > 1: tăng contrast"
        if alpha > 1
        else "0 < α < 1: giảm contrast"
        if alpha < 1
        else "α = 1: giữ contrast"
    )
    messages.append(
        "β > 0: sáng hơn"
        if beta > 0
        else "β < 0: tối hơn"
        if beta < 0
        else "β = 0: giữ brightness"
    )
    if np.array_equal(image, result):
        st.info(
            "Histogram chưa đổi vì phép biến đổi hiện tại là đồng nhất: "
            "α = 1 và β = 0 nên I_out = I_in. Hãy thử β = +50 hoặc α = 1.5."
        )
    else:
        st.info(
            "  ·  ".join(messages)
            + "  ·  Công thức được áp dụng độc lập lên từng pixel."
        )


def gamma_demo(image_rgb: np.ndarray) -> None:
    page_intro("3 · Gamma Correction", "Quan sát đồng thời ảnh và hàm ánh xạ phi tuyến.")
    section_label("1 · PARAMETER")
    mode = st.radio(
        "Không gian xử lý",
        ("Grayscale intensity", "RGB từng kênh"),
        horizontal=True,
        key="gamma_mode",
    )
    image = to_gray(image_rgb) if mode == "Grayscale intensity" else image_rgb
    if mode == "RGB từng kênh":
        st.caption("Cùng một hàm gamma được áp dụng độc lập lên từng kênh R, G, B.")

    p1, p2, p3, _ = st.columns([1, 1, 1, 3])
    p1.button("γ = 0.5", on_click=set_gamma, args=(0.5,), width="stretch")
    p2.button("γ = 1.0", on_click=set_gamma, args=(1.0,), width="stretch")
    p3.button("γ = 2.0", on_click=set_gamma, args=(2.0,), width="stretch")
    gamma_control, intensity_control = st.columns(2, gap="large")
    gamma = gamma_control.slider("γ", 0.1, 5.0, 1.0, 0.05, key="gamma_value")
    sample_intensity = intensity_control.slider(
        "Input intensity mẫu"
        if mode == "Grayscale intensity"
        else "Giá trị kênh mẫu (R/G/B)",
        0,
        255,
        100,
        key="gamma_sample_intensity",
    )
    sample_output = 255.0 * (sample_intensity / 255.0) ** gamma
    result = gamma_correct(image, gamma)

    section_label("2 · INPUT → MAPPING → OUTPUT")
    original, curve, output = st.columns([1, 1.1, 1], gap="large")
    with original:
        show_image(image, "ORIGINAL")
    with curve:
        st.plotly_chart(
            gamma_curve_figure(gamma, sample_intensity), config=PLOT_CONFIG, theme=None
        )
        st.latex(rf"I_{{out}}=255\left(\frac{{I_{{in}}}}{{255}}\right)^{{{gamma:.2f}}}")
        st.metric(
            "Điểm đang đánh dấu: I_in → I_out",
            f"{sample_intensity} → {sample_output:.1f}",
        )
    with output:
        show_image(result, "RESULT")

    section_label("3 · WHAT CHANGED?")
    histogram_mode = "Grayscale" if mode == "Grayscale intensity" else "RGB"
    h1, h2 = st.columns(2, gap="large")
    h1.plotly_chart(
        histogram_figure(image, histogram_mode, "Before"), config=PLOT_CONFIG, theme=None
    )
    h2.plotly_chart(
        histogram_figure(result, histogram_mode, "After"), config=PLOT_CONFIG, theme=None
    )
    if gamma < 1:
        st.info("γ < 1 → khuếch đại vùng tối → ảnh thường sáng hơn.")
    elif gamma > 1:
        st.info("γ > 1 → nén vùng tối → ảnh thường tối hơn.")
    else:
        st.info("γ = 1 → ánh xạ đồng nhất, ảnh không đổi.")


def histogram_demo(image_rgb: np.ndarray) -> None:
    page_intro("4 · Histogram & Contrast", "Histogram nói gì về phân bố cường độ của ảnh?")
    section_label("1 · PARAMETER")
    c1, c2 = st.columns(2)
    hist_mode = c1.radio("Histogram", ("Grayscale", "R / G / B"), horizontal=True, key="hist_mode")
    operation = c2.radio(
        "Xử lý", ("Original", "Histogram Equalization", "CLAHE"), horizontal=True, key="hist_operation"
    )
    clip_limit, tile_size = 2.0, 8
    if operation == "CLAHE":
        p1, p2 = st.columns(2)
        clip_limit = p1.slider("clipLimit", 0.5, 8.0, 2.0, 0.1)
        tile_size = p2.slider("tileGridSize", 2, 16, 8, 1)

    if operation == "Histogram Equalization":
        processed = equalize_histogram(image_rgb)
    elif operation == "CLAHE":
        processed = apply_clahe(image_rgb, clip_limit, tile_size)
    else:
        processed = image_rgb.copy()

    process_description = {
        "Original": "Không biến đổi — dùng làm mốc so sánh.",
        "Histogram Equalization": "Một ánh xạ CDF chung được áp dụng lên toàn ảnh.",
        "CLAHE": f"Chia ảnh thành ô {tile_size}×{tile_size}, giới hạn khuếch đại ở {clip_limit:.1f}.",
    }[operation]
    st.info(f"PROCESSING · {operation}: {process_description}")

    section_label("2 · INPUT → HISTOGRAM → PROCESSING → OUTPUT → NEW HISTOGRAM")
    mode = "RGB" if hist_mode == "R / G / B" else "Grayscale"
    a, b, c, d = st.columns([1, 1.15, 1, 1.15], gap="medium")
    with a:
        show_image(image_rgb, "ORIGINAL")
    with b:
        st.plotly_chart(
            histogram_figure(image_rgb, mode, "Histogram"), config=PLOT_CONFIG, theme=None
        )
    with c:
        show_image(processed, operation.upper())
    with d:
        st.plotly_chart(
            histogram_figure(processed, mode, "New Histogram"), config=PLOT_CONFIG, theme=None
        )

    section_label("3 · WHAT CHANGED?")
    st.markdown(
        """
        - Tập trung về bên trái thường gợi ý ảnh tối; về bên phải thường gợi ý ảnh sáng.

        - Phân bố hẹp thường đi cùng contrast thấp; trải rộng thường đi cùng contrast cao.

        - Đây là dấu hiệu trực quan, không phải luật tuyệt đối cho mọi ảnh.
        """
    )
    if operation == "Histogram Equalization":
        st.info("Equalization dùng CDF để tạo một ánh xạ cường độ chung cho toàn ảnh.")
    elif operation == "CLAHE":
        st.info("CLAHE xử lý theo từng ô nhỏ và giới hạn khuếch đại để tăng contrast cục bộ.")


def noise_filtering_demo(image_rgb: np.ndarray) -> None:
    page_intro(
        "5 · Noise & Filtering",
        "So sánh loại nhiễu và lý do Gaussian/Median phản ứng khác nhau.",
    )
    reset_reveal_for_new_image("noise", image_rgb)
    if "noise_seed" not in st.session_state:
        st.session_state.noise_seed = 2025

    section_label("1 · PARAMETER")
    noise_type = st.radio(
        "Noise",
        ("Gaussian noise", "Salt-and-pepper noise"),
        horizontal=True,
        key="noise_type",
        on_change=hide_noise_result,
    )
    noise_level, new_noise = st.columns([3, 1])
    if noise_type == "Gaussian noise":
        sigma_noise = noise_level.slider(
            "σ_noise",
            0.0,
            60.0,
            20.0,
            1.0,
            key="noise_sigma",
            on_change=hide_noise_result,
        )
        noisy = add_gaussian_noise(image_rgb, sigma_noise, st.session_state.noise_seed)
    else:
        density = noise_level.slider(
            "Mật độ impulse",
            0.0,
            0.20,
            0.08,
            0.01,
            key="noise_density",
            on_change=hide_noise_result,
        )
        noisy = add_salt_pepper_noise(
            image_rgb, density, st.session_state.noise_seed
        )
    new_noise.button(
        "Generate new noise",
        on_click=generate_new_noise,
        width="stretch",
    )
    st.caption(
        f"Seed = {st.session_state.noise_seed}. Cùng seed + cùng tham số → cùng ảnh nhiễu."
    )

    gaussian_controls, median_controls = st.columns(2, gap="large")
    gaussian_kernel = gaussian_controls.selectbox(
        "Gaussian kernel size",
        (3, 5, 7),
        index=1,
        key="noise_gaussian_kernel",
        on_change=hide_noise_result,
    )
    gaussian_sigma = gaussian_controls.slider(
        "Gaussian filter σ",
        0.5,
        4.0,
        1.2,
        0.1,
        key="noise_filter_sigma",
        on_change=hide_noise_result,
    )
    median_kernel = median_controls.selectbox(
        "Median kernel size",
        (3, 5, 7),
        index=1,
        key="noise_median_kernel",
        on_change=hide_noise_result,
    )
    median_controls.caption(
        "Median là bộ lọc phi tuyến: output là trung vị của vùng lân cận, "
        "không phải tổng trọng số như convolution."
    )

    section_label("2 · INPUT → NOISY IMAGE")
    original_col, noisy_col = st.columns(2, gap="large")
    with original_col:
        show_image(image_rgb, "ORIGINAL")
    with noisy_col:
        show_image(noisy, noise_type.upper())

    section_label("3 · DỰ ĐOÁN TRƯỚC KHI XEM KẾT QUẢ")
    if noise_type == "Salt-and-pepper noise":
        question = "Bạn dự đoán Gaussian hay Median sẽ xử lý các pixel 0/255 tốt hơn?"
    else:
        question = "Filter nào sẽ làm ảnh mượt hơn? Điều gì xảy ra với cạnh?"
    st.markdown(f'<div class="note"><b>{question}</b></div>', unsafe_allow_html=True)
    reveal = st.checkbox(
        "Đã dự đoán — hiển thị kết quả lọc",
        key="noise_reveal",
    )
    if not reveal:
        st.caption("Kết quả lọc đang được ẩn để lớp dự đoán trước.")
        return

    gaussian_result = gaussian_filter(noisy, gaussian_kernel, gaussian_sigma)
    median_result = median_filter(noisy, median_kernel)
    section_label("4 · FILTERED OUTPUT")
    gaussian_col, median_col = st.columns(2, gap="large")
    with gaussian_col:
        show_image(
            gaussian_result,
            f"GAUSSIAN FILTER · {gaussian_kernel}×{gaussian_kernel}, σ={gaussian_sigma:.1f}",
        )
    with median_col:
        show_image(median_result, f"MEDIAN FILTER · {median_kernel}×{median_kernel}")

    section_label("5 · WHAT SHOULD WE NOTICE?")
    if noise_type == "Gaussian noise":
        st.info(
            "Gaussian smoothing thường giảm nhiễu ngẫu nhiên nhưng đồng thời làm mờ cạnh. "
            "Median cũng có thể giảm một phần noise, nhưng không phải lúc nào cũng tối ưu."
        )
    else:
        st.info(
            "Gaussian làm pixel cực trị lan sang láng giềng; Median thường loại impulse "
            "outlier tốt hơn trong trường hợp salt-and-pepper thưa."
        )

    with st.expander("Vì sao Median chịu outlier tốt?"):
        neighborhood = np.array(
            [[82, 80, 255], [81, 79, 83], [80, 82, 81]], dtype=np.float32
        )
        gaussian_weights = np.array(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32
        ) / 16.0
        weighted = float(np.sum(neighborhood * gaussian_weights))
        st.code(str(neighborhood.astype(int)), language=None)
        st.write(
            f"Mean = {neighborhood.mean():.1f} · Gaussian weighted = {weighted:.1f} · "
            f"Median = {np.median(neighborhood):.1f}"
        )
        st.caption(
            "Giá trị 255 kéo mean/tổng trọng số lên, trong khi median vẫn chọn giá trị "
            "ở giữa sau khi sắp xếp."
        )


def _format_pixel_value(value: object, decimals: int = 1) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{float(array):.{decimals}f}"
    return "[" + ", ".join(f"{float(item):.{decimals}f}" for item in array) + "]"


def geometric_transform_demo(image_rgb: np.ndarray) -> None:
    page_intro(
        "6 · Geometric Transform",
        "Tọa độ pixel thay đổi; inverse warping tìm source cho từng output pixel.",
    )
    reset_reveal_for_new_image("geometry", image_rgb)
    height, width = image_rgb.shape[:2]

    st.info(
        "QUY ƯỚC · x = column, y = row, gốc (0, 0) ở góc trên-trái. "
        "Point operator giữ vị trí; geometric transform thay đổi tọa độ."
    )
    section_label("1 · PARAMETER")
    operation_col, interpolation_col = st.columns(2, gap="large")
    operation = operation_col.radio(
        "Operation",
        ("Rotate", "Scale", "Translate"),
        horizontal=True,
        key="geometry_operation",
        on_change=hide_geometry_result,
    )
    interpolation = interpolation_col.radio(
        "Interpolation",
        ("Nearest Neighbor", "Bilinear"),
        horizontal=True,
        key="geometry_interpolation",
        on_change=hide_geometry_result,
    )

    if operation == "Rotate":
        angle = st.slider(
            "Góc quay quanh tâm (degree)",
            -180,
            180,
            25,
            1,
            key="geometry_angle",
            on_change=hide_geometry_result,
        )
        matrix = build_transform_matrix(
            (height, width), operation, angle_degrees=angle
        )
        prediction = "Pixel output có luôn ánh xạ đúng vào một pixel nguyên của ảnh gốc không?"
        composition = "M = T_center_back · R · T_center"
    elif operation == "Scale":
        scale = st.slider(
            "Tỉ lệ quanh tâm",
            0.5,
            2.0,
            1.35,
            0.05,
            key="geometry_scale",
            on_change=hide_geometry_result,
        )
        matrix = build_transform_matrix((height, width), operation, scale=scale)
        prediction = "Nếu phóng to bằng nearest neighbor, điều gì xảy ra với các pixel?"
        composition = "M = T_center_back · S · T_center"
    else:
        translation_x, translation_y = st.columns(2)
        tx = translation_x.slider(
            "tx · pixel theo chiều rộng",
            -width,
            width,
            0,
            1,
            key="geometry_tx",
            on_change=hide_geometry_result,
        )
        ty = translation_y.slider(
            "ty · pixel theo chiều cao",
            -height,
            height,
            0,
            1,
            key="geometry_ty",
            on_change=hide_geometry_result,
        )
        matrix = build_transform_matrix((height, width), operation, tx=tx, ty=ty)
        prediction = "Sau khi tịnh tiến, những vùng output nào không có source pixel?"
        composition = "M = T(tx, ty)"

    section_label("2 · INTERMEDIATE STEP — HOMOGENEOUS MATRIX")
    matrix_col, mapping_col = st.columns([1, 1.35], gap="large")
    with matrix_col:
        st.dataframe(
            np.round(matrix, 4),
            width="stretch",
            hide_index=True,
            column_config={0: "x", 1: "y", 2: "translation"},
        )
        st.caption(f"Forward matrix thực sự dùng để warp · {composition}")
    with mapping_col:
        st.latex(
            r"\begin{bmatrix}x'\\y'\\1\end{bmatrix}"
            r"=M\begin{bmatrix}x\\y\\1\end{bmatrix},\qquad "
            r"\begin{bmatrix}x\\y\\1\end{bmatrix}"
            r"=M^{-1}\begin{bmatrix}x'\\y'\\1\end{bmatrix}"
        )
        st.caption(
            "Full image dùng M⁻¹ để tạo source map rồi cv2.remap; output canvas giữ nguyên kích thước input."
        )

    section_label("3 · DỰ ĐOÁN TRƯỚC KHI XEM KẾT QUẢ")
    st.markdown(f'<div class="note"><b>{prediction}</b></div>', unsafe_allow_html=True)
    reveal = st.checkbox(
        "Đã dự đoán — hiển thị kết quả transform",
        key="geometry_reveal",
    )
    if not reveal:
        st.caption("Ảnh output đang được ẩn để lớp dự đoán trước.")
        return

    transformed = warp_inverse(image_rgb, matrix, interpolation)
    section_label("4 · OUTPUT — INVERSE WARPING")
    original_col, transformed_col = st.columns(2, gap="large")
    with original_col:
        show_image(image_rgb, "ORIGINAL")
    with transformed_col:
        show_image(transformed, f"{operation.upper()} · {interpolation}")

    section_label("5 · INSPECT ONE OUTPUT PIXEL")
    coordinate_col, result_col = st.columns([1, 2], gap="large")
    output_x = coordinate_col.slider("x' (output column)", 0, width - 1, width // 2)
    output_y = coordinate_col.slider("y' (output row)", 0, height - 1, height // 2)
    source_x, source_y = inverse_source_coordinate(
        matrix, output_x, output_y
    )
    result_col.markdown(
        f"**Output:** ({output_x}, {output_y})  → M⁻¹ →  "
        f"**Source:** ({source_x:.3f}, {source_y:.3f})"
    )

    if interpolation == "Nearest Neighbor":
        (nearest_x, nearest_y), value = nearest_sample(
            image_rgb, source_x, source_y
        )
        if value is None:
            result_col.warning(
                f"Nearest source = ({nearest_x}, {nearest_y}) nằm ngoài ảnh → border đen."
            )
        else:
            result_col.write(
                f"Nearest source pixel = ({nearest_x}, {nearest_y}) · "
                f"RGB = {_format_pixel_value(value, 0)}"
            )
            result_col.caption(
                "Nearest lấy một pixel gần nhất; khi phóng to thường tạo khối pixel rõ."
            )
    else:
        details = bilinear_sample_details(image_rgb, source_x, source_y)
        rows = []
        labels = ("Q11", "Q21", "Q12", "Q22")
        for label, coordinate, value in zip(
            labels, details["coordinates"], details["values"]
        ):
            rows.append(
                {
                    "Neighbor": label,
                    "(x, y)": str(coordinate),
                    "RGB": _format_pixel_value(value, 0),
                }
            )
        result_col.dataframe(rows, width="stretch", hide_index=True)
        result_col.write(
            f"dx = {details['dx']:.3f}, dy = {details['dy']:.3f} · "
            f"R1 = {_format_pixel_value(details['r1'])} · "
            f"R2 = {_format_pixel_value(details['r2'])}"
        )
        result_col.write(
            f"P = (1-dy)R1 + dyR2 = **{_format_pixel_value(details['value'])}** · "
            f"output uint8 thực tế = {_format_pixel_value(transformed[output_y, output_x], 0)}"
        )
        result_col.caption(
            "Các neighbor ngoài source domain được xem là 0 vì border mode đang là constant black."
        )

    with st.expander("Vì sao dùng inverse warping?"):
        demo_shape = (32, 32)
        demo_matrix = centered_rotation_scale_matrix(
            demo_shape, angle_degrees=18.0, scale=1.25
        )
        forward_hits = forward_mapping_occupancy(demo_shape, demo_matrix)
        inverse_coverage = warp_inverse(
            np.full(demo_shape, 255, dtype=np.uint8),
            demo_matrix,
            "Nearest Neighbor",
        )
        forward_col, inverse_col = st.columns(2)
        with forward_col:
            show_image(forward_hits, "FORWARD · ô trắng đã được source chạm tới")
        with inverse_col:
            show_image(inverse_coverage, "INVERSE · mỗi destination tự tìm source")
        st.caption(
            "Forward source → destination có thể bỏ sót ô bên trong (hole). Inverse hỏi source "
            "cho từng destination nên tránh loại hole do sampling này. Vùng đen ngoài biên vẫn "
            "tồn tại khi destination ánh xạ ra ngoài source domain."
        )


def fourier_demo(image_rgb: np.ndarray) -> None:
    page_intro("7 · Fourier Transform", "Từ miền không gian sang miền tần số, lọc, rồi tái tạo ảnh.")
    reset_fourier_reveal_for_new_image(image_rgb)
    gray = to_gray(image_rgb)
    fft, shifted = compute_fft(gray)
    raw_spectrum = magnitude_spectrum(fft)
    shifted_linear = linear_magnitude_spectrum(shifted)
    shifted_spectrum = magnitude_spectrum(shifted)

    section_label("1 · FFT PIPELINE")
    first_left, first_right = st.columns(2, gap="large")
    with first_left:
        show_image(gray, "1 · ORIGINAL GRAYSCALE")
    with first_right:
        show_image(raw_spectrum, "2 · FFT (DC ở góc)")
    second_left, second_right = st.columns(2, gap="large")
    with second_left:
        show_image(shifted_linear, "3 · FFTSHIFT |F_shift| (linear)")
    with second_right:
        show_image(shifted_spectrum, "4 · LOG MAGNITUDE (display)")
    with st.expander("Xem ba dòng NumPy của pipeline"):
        st.code(
            "F = np.fft.fft2(image)\nF_shift = np.fft.fftshift(F)\nspectrum = np.log1p(np.abs(F_shift))",
            language="python",
        )

    section_label("2 · PARAMETER — FREQUENCY FILTER")
    filter_type = st.radio(
        "Filter",
        (
            "None",
            "Ideal Low-pass",
            "Gaussian Low-pass",
            "Ideal High-pass",
            "Compare Ideal vs Gaussian",
        ),
        horizontal=True,
        key="fourier_filter",
        on_change=hide_fourier_result,
    )
    max_radius = max(1, min(gray.shape) // 2)
    default_width = max(1, min(50, max_radius // 3))
    radius = default_width
    sigma_f = float(default_width)
    if filter_type in ("Ideal Low-pass", "Ideal High-pass"):
        radius = st.slider(
            "Cutoff radius R",
            1,
            max_radius,
            default_width,
            key="fourier_radius",
            on_change=hide_fourier_result,
        )
    elif filter_type == "Gaussian Low-pass":
        sigma_f = st.slider(
            "Gaussian frequency width σ_f",
            1.0,
            float(max_radius),
            float(default_width),
            1.0,
            key="fourier_sigma",
            on_change=hide_fourier_result,
        )
    elif filter_type == "Compare Ideal vs Gaussian":
        ideal_control, gaussian_control = st.columns(2, gap="large")
        radius = ideal_control.slider(
            "Ideal cutoff R",
            1,
            max_radius,
            default_width,
            key="fourier_radius",
            on_change=hide_fourier_result,
        )
        sigma_f = gaussian_control.slider(
            "Gaussian σ_f",
            1.0,
            float(max_radius),
            float(default_width),
            1.0,
            key="fourier_sigma",
            on_change=hide_fourier_result,
        )
        st.caption(
            "R và σ_f đều điều khiển độ rộng vùng tần số thấp, nhưng không phải cùng một định nghĩa cutoff."
        )

    section_label("3 · DỰ ĐOÁN TRƯỚC KHI XEM KẾT QUẢ")
    if filter_type == "Ideal High-pass":
        question = "Nếu tăng cutoff radius, ảnh tái tạo sẽ thay đổi thế nào?"
        answer = (
            "Radius lớn hơn che vùng trung tâm lớn hơn → bỏ thêm tần số thấp "
            "→ kết quả thiên về cạnh/texture hơn."
        )
    elif filter_type == "Ideal Low-pass":
        question = "Nếu giảm cutoff radius, ảnh tái tạo sẽ thay đổi thế nào?"
        answer = "Cutoff nhỏ hơn giữ ít tần số hơn → mất thêm chi tiết cao tần → ảnh mờ hơn."
    elif filter_type == "Gaussian Low-pass":
        question = "Nếu giảm σ_f, ảnh tái tạo và cạnh sắc sẽ thay đổi thế nào?"
        answer = "σ_f nhỏ hơn làm pass-band hẹp hơn → ảnh mượt hơn và cạnh chuyển tiếp rộng hơn."
    elif filter_type == "Compare Ideal vs Gaussian":
        question = "Với độ rộng gần nhau, Ideal và Gaussian khác gì quanh một cạnh sắc?"
        answer = (
            "Ideal có cutoff đột ngột nên dễ tạo dao động Gibbs/ringing. Gaussian chuyển tiếp "
            "mượt nên thường giảm mạnh ringing do cutoff đột ngột."
        )
    else:
        question = "Mask toàn trắng sẽ làm thay đổi ảnh tái tạo hay không?"
        answer = "Không; mọi hệ số Fourier được giữ lại, sai khác chỉ ở mức số học."
    st.markdown(f'<div class="note"><b>{question}</b></div>', unsafe_allow_html=True)
    reveal = st.checkbox(
        "Đã dự đoán — hiển thị mask và kết quả",
        key="fourier_reveal",
    )
    if not reveal:
        st.caption("Kết quả đang được ẩn để giảng viên có thể dừng lại và hỏi lớp.")
        return

    if filter_type == "Compare Ideal vs Gaussian":
        ideal_mask = create_lowpass_mask(gray.shape, radius)
        gaussian_mask = create_gaussian_lowpass_mask(gray.shape, sigma_f)
        ideal_masked = apply_frequency_mask(shifted, ideal_mask)
        gaussian_masked = apply_frequency_mask(shifted, gaussian_mask)
        ideal_reconstructed = reconstruct_image(ideal_masked)
        gaussian_reconstructed = reconstruct_image(gaussian_masked)

        section_label("4 · INPUT → ORIGINAL SPECTRUM → TWO FILTER BRANCHES")
        input_col, spectrum_col = st.columns(2, gap="large")
        with input_col:
            show_image(gray, "INPUT · GRAYSCALE")
        with spectrum_col:
            show_image(shifted_spectrum, "ORIGINAL SPECTRUM · log display")

        ideal_col, gaussian_col = st.columns(2, gap="large")
        with ideal_col:
            st.markdown("#### IDEAL LOW-PASS")
            show_image((ideal_mask * 255).astype(np.uint8), f"IDEAL MASK · R={radius}")
            show_image(magnitude_spectrum(ideal_masked), "IDEAL MASKED SPECTRUM")
            show_image(
                reconstruction_for_display(ideal_reconstructed),
                "IDEAL IFFT · clip [0,255]",
            )
        with gaussian_col:
            st.markdown("#### GAUSSIAN LOW-PASS")
            show_image(
                np.rint(gaussian_mask * 255).astype(np.uint8),
                f"GAUSSIAN MASK · σ_f={sigma_f:.1f}",
            )
            show_image(magnitude_spectrum(gaussian_masked), "GAUSSIAN MASKED SPECTRUM")
            show_image(
                reconstruction_for_display(gaussian_reconstructed),
                "GAUSSIAN IFFT · clip [0,255]",
            )
        st.caption(
            f"IFFT thật · Ideal [{ideal_reconstructed.min():.2f}, {ideal_reconstructed.max():.2f}] "
            f"· Gaussian [{gaussian_reconstructed.min():.2f}, {gaussian_reconstructed.max():.2f}]. "
            "Không min–max normalize reconstruction."
        )

        section_label("5 · WHAT CHANGED?")
        st.info(
            "Ideal mask có biên cắt đột ngột nên dễ tạo ringing/Gibbs quanh cạnh sắc. "
            "Gaussian mask không có discontinuity sắc như Ideal LP, nên giảm mạnh ringing "
            "do cutoff đột ngột; mức quan sát còn phụ thuộc ảnh và tham số."
        )
        with st.expander("OPTIONAL · 1D INTENSITY PROFILE"):
            profile_row = st.slider(
                "Hàng ngang y",
                0,
                gray.shape[0] - 1,
                gray.shape[0] // 2,
                key="fourier_profile_row",
            )
            st.plotly_chart(
                intensity_profile_figure(
                    gray.astype(np.float32),
                    ideal_reconstructed,
                    gaussian_reconstructed,
                    profile_row,
                ),
                config=PLOT_CONFIG,
                theme=None,
            )
            edge_strength = float(
                np.max(np.abs(np.diff(gray[profile_row].astype(np.float32))))
            )
            if edge_strength < 40:
                st.warning(
                    "Hàng này chưa có cạnh đủ sắc để kết luận về ringing. Hãy chọn hàng khác "
                    "hoặc dùng sample Ringing test pattern."
                )
            else:
                st.caption(
                    "Quan sát dao động/overshoot gần vị trí cường độ đổi đột ngột; profile giữ "
                    "giá trị IFFT thật nên có thể vượt [0,255]."
                )
        with st.expander("Show explanation"):
            st.write(answer)
        return

    if filter_type == "Ideal Low-pass":
        mask = create_lowpass_mask(gray.shape, radius)
    elif filter_type == "Gaussian Low-pass":
        mask = create_gaussian_lowpass_mask(gray.shape, sigma_f)
    elif filter_type == "Ideal High-pass":
        mask = create_highpass_mask(gray.shape, radius)
    else:
        mask = np.ones(gray.shape, dtype=np.float32)

    masked = apply_frequency_mask(shifted, mask)
    filtered_spectrum = magnitude_spectrum(masked)
    reconstructed = reconstruct_image(masked)
    signed_display = filter_type == "Ideal High-pass"
    reconstructed_display = reconstruction_for_display(
        reconstructed, signed=signed_display
    )
    reconstruction_caption = (
        "IFFT DISPLAY · signed: 0 → xám giữa"
        if signed_display
        else "RECONSTRUCTED · chỉ clip [0,255]"
    )

    section_label("4 · INPUT → SPECTRUM → MASK → MODIFIED SPECTRUM → IFFT")
    first, second, third = st.columns(3, gap="medium")
    with first:
        show_image(gray, "1 · SPATIAL IMAGE")
    with second:
        show_image(shifted_spectrum, "2 · ORIGINAL SPECTRUM")
    with third:
        show_image(np.rint(mask * 255).astype(np.uint8), "3 · FREQUENCY MASK")
    fourth, fifth = st.columns(2, gap="large")
    with fourth:
        show_image(filtered_spectrum, "4 · MASKED SPECTRUM")
    with fifth:
        show_image(reconstructed_display, f"5 · {reconstruction_caption}")
    st.caption(
        f"IFFT thật: min = {reconstructed.min():.2f}, max = {reconstructed.max():.2f}. "
        + (
            "High-pass dùng ánh xạ đối xứng chỉ để hiển thị dấu; dữ liệu IFFT không bị normalize."
            if signed_display
            else "Không min–max normalize; chỉ clip về miền hiển thị 8-bit."
        )
    )

    section_label("5 · WHAT CHANGED?")
    if filter_type == "Ideal Low-pass":
        st.info("Ideal LP giữ vùng tròn gần tâm; cutoff nhỏ hơn làm ảnh mờ hơn và biên mask vẫn cắt đột ngột.")
    elif filter_type == "Gaussian Low-pass":
        st.info("Gaussian LP giảm dần theo khoảng cách tới tâm; σ_f nhỏ hơn làm ảnh mượt hơn mà không tạo biên mask nhị phân.")
    elif filter_type == "Ideal High-pass":
        st.info("Loại vùng gần tâm → giữ tần số cao → nhấn mạnh thay đổi nhanh, cạnh và texture.")
    else:
        st.info("Mask toàn trắng giữ mọi tần số → ảnh tái tạo không đổi.")

    with st.expander("Show explanation"):
        st.write(answer)


KERNELS: dict[str, tuple[list[list[str]], str, str]] = {
    "Identity": (
        [["0", "0", "0"], ["0", "1", "0"], ["0", "0", "0"]],
        "Giữ nguyên ảnh.",
        "Chỉ pixel trung tâm được truyền sang output.",
    ),
    "Box Blur": (
        [["1/9", "1/9", "1/9"], ["1/9", "1/9", "1/9"], ["1/9", "1/9", "1/9"]],
        "Làm mờ bằng trung bình đều.",
        "Cạnh bị chuyển tiếp dần khi cửa sổ trượt qua.",
    ),
    "Gaussian Blur": (
        [["1/16", "2/16", "1/16"], ["2/16", "4/16", "2/16"], ["1/16", "2/16", "1/16"]],
        "Làm mờ có trọng số, σ ≈ 0.85.",
        "Pixel gần tâm đóng góp nhiều hơn; tổng kernel bằng 1.",
    ),
    "Sharpen": (
        [["0", "-1", "0"], ["-1", "5", "-1"], ["0", "-1", "0"]],
        "Tăng cường chi tiết và cạnh.",
        "Tâm lớn, láng giềng âm → tăng khác biệt cục bộ.",
    ),
    "Edge Detection": (
        [["-1", "-1", "-1"], ["-1", "8", "-1"], ["-1", "-1", "-1"]],
        "Phản hồi mạnh tại thay đổi cường độ.",
        "Vùng phẳng gần triệt tiêu vì tổng kernel bằng 0.",
    ),
    "Sobel X": (
        [["-1", "0", "1"], ["-2", "0", "2"], ["-1", "0", "1"]],
        "Xấp xỉ đạo hàm theo trục x.",
        "Làm nổi bật cạnh đứng.",
    ),
    "Sobel Y": (
        [["-1", "-2", "-1"], ["0", "0", "0"], ["1", "2", "1"]],
        "Xấp xỉ đạo hàm theo trục y.",
        "Làm nổi bật cạnh ngang.",
    ),
    "Emboss": (
        [["-2", "-1", "0"], ["-1", "1", "1"], ["0", "1", "2"]],
        "Tạo cảm giác nổi theo một hướng.",
        "Đổi hướng kernel sẽ đổi hướng sáng–tối của hiệu ứng.",
    ),
}


def kernel_text(matrix: list[list[str]]) -> str:
    return "[ " + "\n  ".join("  ".join(row) for row in matrix) + " ]"


def convolution_library() -> None:
    intro, animation = st.columns([1.35, 1], gap="large", vertical_alignment="center")
    with intro:
        page_intro(
            "8 · Convolution Kernel Library",
            "Chọn kernel, đọc các hệ số, rồi thử trực tiếp trên Setosa.",
        )
        st.link_button(
            "Mở Interactive convolution demo ↗",
            "https://setosa.io/ev/image-kernels/",
            type="primary",
        )
        st.caption("Kernel Gaussian 3×3 bên dưới dùng đúng hệ số trong Chapter 1.tex.")
    with animation:
        st.image(
            str(SLIDING_WINDOW_GIF),
            caption="Cửa sổ xử lý quét ảnh từ trái sang phải, trên xuống dưới.",
            width="stretch",
        )

    section_label("KERNEL 3 × 3")
    choice = st.selectbox("Chọn kernel", tuple(KERNELS))
    matrix, effect, observe = KERNELS[choice]
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.code(kernel_text(matrix), language=None)
    with right:
        st.markdown(f"**Tác dụng:** {effect}")
        st.markdown(f"**Nên quan sát:** {observe}")
        st.markdown(
            '<div class="note"><b>Dự đoán trước:</b> vùng phẳng và vùng có cạnh sẽ phản ứng khác nhau thế nào?</div>',
            unsafe_allow_html=True,
        )

    section_label("QUICK REFERENCE")
    columns = st.columns(4)
    for index, (name, (values, _, _)) in enumerate(KERNELS.items()):
        with columns[index % 4]:
            with st.expander(name):
                st.code(kernel_text(values), language=None)


image_rgb, image_name = image_source()
st.sidebar.divider()
st.sidebar.header("BÀI DEMO")
page = st.sidebar.radio(
    "Điều hướng",
    (
        "1. Pixel Explorer",
        "2. Brightness & Contrast",
        "3. Gamma",
        "4. Histogram",
        "5. Noise & Filtering",
        "6. Geometric Transform",
        "7. Fourier",
        "8. Convolution Kernels",
    ),
    label_visibility="collapsed",
    key="nav_page",
)
st.sidebar.divider()
st.sidebar.caption(f"Đang dùng: {image_name}")
st.sidebar.caption("Chapter 1 · Xử lý ảnh & Thị giác máy tính")

if page.startswith("1"):
    pixel_explorer(image_rgb)
elif page.startswith("2"):
    brightness_contrast(image_rgb)
elif page.startswith("3"):
    gamma_demo(image_rgb)
elif page.startswith("4"):
    histogram_demo(image_rgb)
elif page.startswith("5"):
    noise_filtering_demo(image_rgb)
elif page.startswith("6"):
    geometric_transform_demo(image_rgb)
elif page.startswith("7"):
    fourier_demo(image_rgb)
else:
    convolution_library()
