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
    create_highpass_mask,
    create_lowpass_mask,
    linear_magnitude_spectrum,
    magnitude_spectrum,
    reconstruct_image,
    reconstruction_for_display,
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
from utils.visualization import PLOT_CONFIG, gamma_curve_figure, histogram_figure


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "assets" / "samples"
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
      .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px;}
      html, body, [class*="css"] {font-size: 17px;}
      h1 {font-size: 2.25rem !important; color: #123b5d; margin-bottom: .25rem !important;}
      h2 {font-size: 1.55rem !important; color: #155e75;}
      h3 {font-size: 1.15rem !important; letter-spacing: .035em; text-transform: uppercase; color: #475569;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .flow-card {background: #f0f7fb; border: 1px solid #cbdde8; border-radius: 12px;
                  padding: 1rem; min-height: 150px; display: flex; flex-direction: column;
                  align-items: center; justify-content: center; text-align: center;}
      .flow-label {font-weight: 800; color: #155e75; letter-spacing: .07em; font-size: .82rem;}
      .formula {font-size: 1.32rem; font-weight: 700; color: #c2410c; margin: .8rem 0;}
      .note {background: #fff7ed; border-left: 5px solid #f97316; padding: .75rem 1rem;
             border-radius: 6px; font-size: 1.02rem;}
      .arrow {text-align: center; font-size: 1.8rem; color: #0e7490; font-weight: 800;}
      [data-testid="stSidebar"] {min-width: 315px; max-width: 315px;}
      [data-testid="stSidebar"] h2 {font-size: 1.15rem !important;}
      div[data-testid="stImage"] img {border-radius: 8px;}
    </style>
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
    st.title(title)
    st.caption(purpose)


def section_label(text: str) -> None:
    st.markdown(f"### {text}")


def show_image(image: np.ndarray, caption: str) -> None:
    st.image(image, caption=caption, width="stretch", clamp=True)


def reset_bc() -> None:
    st.session_state.bc_alpha = 1.0
    st.session_state.bc_beta = 0


def set_gamma(value: float) -> None:
    st.session_state.gamma_value = value


def hide_fourier_result() -> None:
    st.session_state.fourier_reveal = False


def reset_fourier_reveal_for_new_image(image: np.ndarray) -> None:
    """Hide Fourier output only when the input pixels actually change."""
    fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
    previous = st.session_state.get("fourier_input_fingerprint")
    if previous is not None and previous != fingerprint:
        st.session_state.fourier_reveal = False
    st.session_state.fourier_input_fingerprint = fingerprint


def pixel_explorer(image_rgb: np.ndarray) -> None:
    page_intro("1 · Pixel Explorer", "Ảnh số là một ma trận các giá trị pixel.")
    mode = st.radio("Biểu diễn", ("RGB", "Grayscale"), horizontal=True, key="pixel_mode")
    image = image_rgb if mode == "RGB" else to_gray(image_rgb)
    height, width = image.shape[:2]
    channels = 3 if image.ndim == 3 else 1

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section_label("1 · INPUT — ORIGINAL IMAGE")
        image_slot = st.empty()
    with right:
        section_label("2 · PIXEL DATA")
        c1, c2 = st.columns(2)
        c1.metric("Kích thước", f"{width} × {height}")
        c2.metric("Số channel", channels)
        c1.metric("dtype", str(image.dtype))
        c2.metric("Min → Max", f"{image.min()} → {image.max()}")
        st.metric("Mean intensity", f"{float(image.mean()):.1f}")
        st.markdown(
            '<div class="note">Mỗi vị trí <b>(x, y)</b> trỏ đến một ô trong ma trận ảnh.</div>',
            unsafe_allow_html=True,
        )

    section_label("3 · INSPECT PIXEL")
    sx, sy, value_box = st.columns([1.1, 1.1, 1], gap="large")
    x = sx.slider("x (cột)", 0, width - 1, width // 2)
    y = sy.slider("y (hàng)", 0, height - 1, height // 2)
    value = image[y, x]
    if image.ndim == 3:
        value_text = f"[{int(value[0])}, {int(value[1])}, {int(value[2])}]"
        value_box.metric(f"Pixel({x}, {y}) = [R, G, B]", value_text)
    else:
        value_box.metric(f"Pixel({x}, {y}) = I", int(value))

    radius = 3
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = image[y0:y1, x0:x1]
    marked_image = mark_pixel_and_patch(image, x, y, radius)
    image_slot.image(
        marked_image,
        caption=f"{mode} · dấu đỏ: pixel ({x}, {y}) · khung vàng: patch",
        width="stretch",
        clamp=True,
    )
    patch_image, patch_values = st.columns([1, 1.5], gap="large")
    with patch_image:
        section_label("4 · PATCH QUANH PIXEL")
        st.image(patch, caption=f"Patch {patch.shape[1]} × {patch.shape[0]}", width=330, clamp=True)
    with patch_values:
        section_label("5 · NHỮNG CON SỐ BÊN TRONG")
        gray_patch = patch if patch.ndim == 2 else to_gray(patch)
        st.dataframe(
            gray_patch,
            width="stretch",
            hide_index=True,
            height=285,
        )
        st.caption("RGB được đổi sang intensity chỉ để bảng 7×7 dễ đọc.")


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
    h1, h2 = st.columns(2, gap="large")
    h1.plotly_chart(
        histogram_figure(image, "Grayscale", "Histogram Before"),
        config=PLOT_CONFIG,
    )
    h2.plotly_chart(
        histogram_figure(result, "Grayscale", "Histogram After"),
        config=PLOT_CONFIG,
    )
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
            gamma_curve_figure(gamma, sample_intensity), config=PLOT_CONFIG
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
    h1.plotly_chart(histogram_figure(image, histogram_mode, "Before"), config=PLOT_CONFIG)
    h2.plotly_chart(histogram_figure(result, histogram_mode, "After"), config=PLOT_CONFIG)
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
        st.plotly_chart(histogram_figure(image_rgb, mode, "Histogram"), config=PLOT_CONFIG)
    with c:
        show_image(processed, operation.upper())
    with d:
        st.plotly_chart(histogram_figure(processed, mode, "New Histogram"), config=PLOT_CONFIG)

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


def fourier_demo(image_rgb: np.ndarray) -> None:
    page_intro("5 · Fourier Transform", "Từ miền không gian sang miền tần số, lọc, rồi tái tạo ảnh.")
    reset_fourier_reveal_for_new_image(image_rgb)
    gray = to_gray(image_rgb)
    fft, shifted = compute_fft(gray)
    raw_spectrum = magnitude_spectrum(fft)
    shifted_linear = linear_magnitude_spectrum(shifted)
    shifted_spectrum = magnitude_spectrum(shifted)

    section_label("1 · FFT PIPELINE")
    a, b, c, d = st.columns(4, gap="medium")
    with a:
        show_image(gray, "1 · ORIGINAL GRAYSCALE")
    with b:
        show_image(raw_spectrum, "2 · FFT (DC ở góc)")
    with c:
        show_image(shifted_linear, "3 · FFTSHIFT |F_shift| (linear)")
    with d:
        show_image(shifted_spectrum, "4 · LOG MAGNITUDE (display)")
    with st.expander("Xem ba dòng NumPy của pipeline"):
        st.code(
            "F = np.fft.fft2(image)\nF_shift = np.fft.fftshift(F)\nspectrum = np.log1p(np.abs(F_shift))",
            language="python",
        )

    section_label("2 · PARAMETER — FREQUENCY FILTER")
    control1, control2 = st.columns([1.2, 2])
    filter_type = control1.radio(
        "Filter",
        ("None", "Ideal Low-pass", "Ideal High-pass"),
        key="fourier_filter",
        on_change=hide_fourier_result,
    )
    max_radius = max(1, min(gray.shape) // 2)
    default_radius = max(1, min(50, max_radius // 3))
    radius = control2.slider(
        "Radius / cutoff frequency",
        1,
        max_radius,
        default_radius,
        key="fourier_radius",
        disabled=filter_type == "None",
        on_change=hide_fourier_result,
    )

    section_label("3 · DỰ ĐOÁN TRƯỚC KHI XEM KẾT QUẢ")
    if filter_type == "Ideal High-pass":
        question = "Nếu tăng cutoff radius, ảnh tái tạo sẽ thay đổi thế nào?"
        answer = (
            "Radius lớn hơn che vùng trung tâm lớn hơn → bỏ thêm tần số thấp "
            "→ ít cấu trúc mượt, kết quả thiên về cạnh/texture hơn."
        )
    elif filter_type == "Ideal Low-pass":
        question = "Nếu giảm cutoff radius, ảnh tái tạo sẽ thay đổi thế nào?"
        answer = "Cutoff nhỏ hơn giữ ít tần số hơn → mất thêm chi tiết cao tần → ảnh mờ hơn."
    else:
        question = "Mask toàn trắng sẽ làm thay đổi ảnh tái tạo hay không?"
        answer = (
            "Không. Mọi hệ số Fourier đều được giữ lại nên IFFT tái tạo ảnh "
            "ban đầu, sai khác chỉ ở mức làm tròn số."
        )
    st.markdown(f'<div class="note"><b>{question}</b></div>', unsafe_allow_html=True)
    reveal = st.checkbox(
        "Đã dự đoán — hiển thị mask và kết quả",
        key="fourier_reveal",
    )
    if not reveal:
        st.caption("Kết quả đang được ẩn để giảng viên có thể dừng lại và hỏi lớp.")
        return

    if filter_type == "Ideal Low-pass":
        mask = create_lowpass_mask(gray.shape, radius)
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
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        show_image(gray, "1 · SPATIAL IMAGE")
    with f2:
        show_image(shifted_spectrum, "2 · ORIGINAL SPECTRUM")
    with f3:
        show_image((mask * 255).astype(np.uint8), "3 · FREQUENCY MASK")
    f4, f5 = st.columns(2, gap="large")
    with f4:
        show_image(filtered_spectrum, "4 · MASKED SPECTRUM")
    with f5:
        show_image(reconstructed_display, f"5 · {reconstruction_caption}")
    st.caption(
        f"IFFT thật: min = {reconstructed.min():.2f}, max = {reconstructed.max():.2f}. "
        + (
            "Ảnh high-pass dùng ánh xạ đối xứng chỉ để hiển thị dấu; dữ liệu IFFT không bị normalize."
            if signed_display
            else "Không min–max normalize; chỉ clip về miền hiển thị 8-bit."
        )
    )

    section_label("5 · WHAT CHANGED?")
    if filter_type == "Ideal Low-pass":
        st.info("Giữ vùng gần tâm → giữ tần số thấp → ảnh mượt hơn và mất chi tiết. Cutoff nhỏ hơn làm ảnh mờ hơn.")
    elif filter_type == "Ideal High-pass":
        st.info("Loại vùng gần tâm → giữ tần số cao → nhấn mạnh thay đổi nhanh, cạnh và texture. Radius lớn hơn loại nhiều cấu trúc chậm hơn.")
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
    "Gaussian Blur (slide)": (
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
    page_intro("6 · Convolution Kernel Library", "Chọn kernel, đọc các hệ số, rồi thử trực tiếp trên Setosa.")
    st.link_button(
        "Mở Interactive convolution demo ↗",
        "https://setosa.io/ev/image-kernels/",
        type="primary",
    )
    st.caption("Kernel Gaussian 3×3 bên dưới dùng đúng hệ số trong Chapter 1.tex.")
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
        "5. Fourier",
        "6. Convolution Kernels",
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
    fourier_demo(image_rgb)
else:
    convolution_library()
