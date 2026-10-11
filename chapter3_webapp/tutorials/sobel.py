"""Guided Sobel story and two real-computation experiment workspaces."""

from __future__ import annotations

from html import escape

import numpy as np
import streamlit as st

from utils.sobel import (DEFAULT_CENTER, KX, KY, PRESET_NAMES, compute_gradient_maps,
                         compute_sobel_at, create_sobel_presets, edit_pixel)
from utils.sobel_visualization import matrix_figure, vector_figure
from utils.visualization import PLOT_CONFIG

STEPS = (
    "Pixels là số", "Chọn patch 3×3", "Gx mask", "Nhân từng cặp",
    "Cộng thành Gx", "Gy trên cùng patch", "Gradient vector", "Magnitude & direction",
    "Trượt kernel", "Các hướng cạnh", "Noise thay đổi gì?", "Tổng kết",
)
QUESTIONS = (
    "Biên dọc tạo response theo trục nào?", "Biên ngang tạo response theo trục nào?",
    "Biên chéo có thể có cả Gx và Gy?", "Vùng phẳng có hướng gradient không?",
    "Đảo tương phản có thể đảo dấu Gx?", "Noise có thể tạo response dù không có biên vật thể?",
)
TAKEAWAYS = (
    "Gx lớn vì intensity tăng từ trái sang phải; gradient vuông góc với biên dọc.",
    "Gy lớn vì intensity tăng từ trên xuống dưới.",
    "Cả hai thành phần đo thay đổi theo hai trục; hướng cục bộ không phải biên toàn vật thể.",
    "M=0 nên hướng không xác định; atan2(0,0) không có ý nghĩa ở đây.",
    "Gx âm: gradient hướng về phía intensity tăng (sang trái).",
    "Sobel phản ứng với thay đổi cục bộ, kể cả noise và texture.",
)


def _init() -> None:
    state = st.session_state
    state.setdefault("sobel_guided_step", 1)
    state.setdefault("sobel_preset", PRESET_NAMES[0])
    state.setdefault("sobel_image", create_sobel_presets()[PRESET_NAMES[0]])
    state.setdefault("sobel_center", DEFAULT_CENTER)
    state.setdefault("sobel_pixel_row", 2)
    state.setdefault("sobel_pixel_col", 3)
    state.setdefault("sobel_edit_value", int(state.sobel_image[state.sobel_pixel_row, state.sobel_pixel_col]))
    state.setdefault("sobel_axis", "Gx")
    state.setdefault("sobel_progress", 0)
    state.setdefault("sobel_reveal_gy", False)
    state.setdefault("sobel_reveal_final", False)
    state.setdefault("sobel_reveal_preset", False)
    state.setdefault("sobel_reveal_noise", False)


def _reset_progress() -> None:
    st.session_state.sobel_progress = 0


def _invalidate() -> None:
    _reset_progress()
    for key in ("sobel_reveal_gy", "sobel_reveal_preset", "sobel_reveal_noise", "sobel_reveal_final",
                "sobel_predict_gy", "sobel_predict_preset", "sobel_predict_noise", "sobel_predict_final"):
        st.session_state.pop(key, None)


def _load_preset() -> None:
    state = st.session_state
    state.sobel_image = create_sobel_presets()[state.sobel_preset]
    state.sobel_center = DEFAULT_CENTER
    row, col = int(state.get("sobel_pixel_row", 2)), int(state.get("sobel_pixel_col", 3))
    state.sobel_edit_value = int(state.sobel_image[row, col])
    _invalidate()


def _select_pixel() -> None:
    state = st.session_state
    state.sobel_edit_value = int(state.sobel_image[state.sobel_pixel_row, state.sobel_pixel_col])


def _edit() -> None:
    state = st.session_state
    try:
        state.sobel_image = edit_pixel(state.sobel_image, int(state.sobel_pixel_row),
                                       int(state.sobel_pixel_col), int(state.sobel_edit_value))
    except ValueError as exc:
        state.sobel_edit_error = str(exc)
        return
    state.sobel_edit_error = ""
    _invalidate()


def _move_center(dr: int, dc: int) -> None:
    state = st.session_state
    row, col = state.sobel_center
    h, w = state.sobel_image.shape
    state.sobel_center = (min(max(row + dr, 1), h - 2),
                          min(max(col + dc, 1), w - 2))
    _invalidate()


def _set_center() -> None:
    state = st.session_state
    state.sobel_center = (int(state.sobel_center_row), int(state.sobel_center_col))
    _invalidate()


def _set_axis() -> None:
    _reset_progress()


def _advance_product(change: int) -> None:
    st.session_state.sobel_progress = min(9, max(0, st.session_state.sobel_progress + change))


def _set_product_progress(value: int) -> None:
    st.session_state.sobel_progress = value


def _guided_move(change: int) -> None:
    st.session_state.sobel_guided_step = min(12, max(1, st.session_state.sobel_guided_step + change))


def _guided_reset() -> None:
    state = st.session_state
    state.sobel_guided_step = 1
    state.sobel_preset = PRESET_NAMES[0]
    _load_preset()
    state.sobel_axis = "Gx"


def _header(title: str, purpose: str) -> None:
    st.markdown('<div class="chapter-kicker">CHAPTER 3 · INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.title(title)
    if not st.session_state.get("sobel_presentation", False):
        st.markdown(f'<div class="hero-copy">{purpose}</div>', unsafe_allow_html=True)


def _progress(step: int) -> None:
    cells = "".join(f'<div class="step {"active" if i == step else "done" if i < step else ""}">{i}<br>{escape(name)}</div>'
                    for i, name in enumerate(STEPS, 1))
    st.markdown(f'<div class="stepper sobel-stepper">{cells}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-progress">Tiến độ học · {step}/12'
                f'<div class="progress-track"><span style="width:{step / 12 * 100:.1f}%"></span></div></div>',
                unsafe_allow_html=True)


def _matrix_html(values: np.ndarray, *, active=None, count=9, color="#123b5d") -> str:
    rows = []
    for row in range(3):
        cells = []
        for col in range(3):
            index = row * 3 + col
            label = str(int(values[row, col])) if index < count else "·"
            selected = " active" if active == index else ""
            cells.append(f'<span class="sobel-cell{selected}">{label}</span>')
        rows.append("".join(cells))
    return f'<div class="sobel-matrix" style="--matrix-color:{color}">{"".join(rows)}</div>'


def _arithmetic(axis: str, progress: int, *, compact=False) -> None:
    state = st.session_state
    result = compute_sobel_at(state.sobel_image, state.sobel_center)
    kernel = KX if axis == "Gx" else KY
    products = result.products_x if axis == "Gx" else result.products_y
    row_sums = result.row_sums_x if axis == "Gx" else result.row_sums_y
    active = min(progress, 8) if progress < 9 else None
    with st.container(key="sobel_panels"):
        a, b, c = st.columns(3, gap="small")
        with a:
            st.markdown("**Patch · 9 pixel thật**")
            st.markdown(_matrix_html(result.patch, active=active, color="#00a7bf"), unsafe_allow_html=True)
        with b:
            st.markdown(f"**{axis} · mask áp trực tiếp**")
            st.markdown(_matrix_html(kernel, active=active,
                                     color="#e85d2a" if axis == "Gx" else "#8152be"),
                        unsafe_allow_html=True)
        with c:
            st.markdown("**Tích từng ô**")
            st.markdown(_matrix_html(products, active=active, count=progress,
                                     color="#14966f"), unsafe_allow_html=True)
    if progress < 9:
        r, c = divmod(progress, 3)
        st.info(f"Cặp tiếp theo · ô ({r},{c}): {int(result.patch[r,c])} × ({int(kernel[r,c]):+d}) "
                f"= {int(products[r,c]):+d}")
    visible = products.ravel()[:progress]
    st.caption(f"Đã tính {progress}/9 tích · tổng đang có = {int(visible.sum()) if visible.size else 0}. "
               "Các ô chưa tính chưa đóng góp vào tổng.")
    if progress == 9:
        st.markdown(f'<div class="formula-box">Hàng: {int(row_sums[0]):+d} '
                    f'+ ({int(row_sums[1]):+d}) + ({int(row_sums[2]):+d}) '
                    f'→ <strong>{axis} = {int(products.sum()):+d}</strong></div>', unsafe_allow_html=True)
    if not compact:
        _operation_controls()


def _operation_controls() -> None:
    with st.container(key="sobel_operations"):
        a, b, c, d = st.columns(4)
        a.button("← Product", on_click=_advance_product, args=(-1,),
                 disabled=st.session_state.sobel_progress == 0, width="stretch")
        b.button("Next Product →", on_click=_advance_product, args=(1,),
                 disabled=st.session_state.sobel_progress == 9, width="stretch")
        c.button("Calculate Remaining", on_click=_set_product_progress, args=(9,), width="stretch")
        d.button("Reset Calculation", on_click=_set_product_progress, args=(0,), width="stretch")


def _image(*, active=None, title="Ảnh grayscale · hàng xuống, cột sang phải") -> None:
    state = st.session_state
    with st.container(key="sobel_image_chart"):
        st.plotly_chart(matrix_figure(state.sobel_image, title, center=state.sobel_center,
                                      patch=True, active=active, layer="Image",
                                      presentation=bool(state.get("sobel_presentation", False))),
                        config=PLOT_CONFIG)


def _results(*, show_vector=True) -> None:
    state = st.session_state
    result = compute_sobel_at(state.sobel_image, state.sobel_center)
    a, b, c, d = st.columns(4)
    a.metric("Gx · signed", f"{result.gx:+d}")
    b.metric("Gy · signed", f"{result.gy:+d}")
    c.metric("M · magnitude", f"{result.magnitude:.2f}")
    d.metric("θ · image coords", "Undefined" if result.theta_degrees is None else f"{result.theta_degrees:.1f}°")
    if show_vector:
        with st.container(key="sobel_vector_chart"):
            st.plotly_chart(vector_figure(result.gx, result.gy,
                                          presentation=bool(state.get("sobel_presentation", False))),
                            config=PLOT_CONFIG)
        st.caption("Image: row/y tăng xuống. Để vẽ trên trục Cartesian y↑, dùng vector (Gx, −Gy). "
                   "Gradient chỉ hướng tăng intensity; edge tangent vuông góc cục bộ với nó.")


def _center_controls() -> None:
    state = st.session_state
    h, w = state.sobel_image.shape
    row, col = state.sobel_center
    with st.container(key="sobel_center_controls"):
        a, b, c, d = st.columns(4)
        a.button("↑ Row", on_click=_move_center, args=(-1, 0), disabled=row == 1, width="stretch")
        b.button("↓ Row", on_click=_move_center, args=(1, 0), disabled=row == h-2, width="stretch")
        c.button("← Col", on_click=_move_center, args=(0, -1), disabled=col == 1, width="stretch")
        d.button("→ Col", on_click=_move_center, args=(0, 1), disabled=col == w-2, width="stretch")
    st.caption(f"Kernel center = ({row}, {col}) · valid rows 1–{h-2}, cols 1–{w-2}; không dùng padding.")


def _preset_controls() -> None:
    st.selectbox("Teaching preset", PRESET_NAMES, key="sobel_preset", on_change=_load_preset)
    index = PRESET_NAMES.index(st.session_state.sobel_preset)
    st.markdown(f'<div class="question">Dự đoán · {escape(QUESTIONS[index])}</div>', unsafe_allow_html=True)
    st.radio("Dự đoán response", ("Gx trội", "Gy trội", "Cả hai", "Bằng 0 / undefined", "Có response vì noise"),
             key="sobel_predict_preset", index=None, horizontal=True)
    st.button("Reveal preset", disabled=st.session_state.get("sobel_predict_preset") is None,
              on_click=lambda: st.session_state.update(sobel_reveal_preset=True))
    if st.session_state.get("sobel_reveal_preset", False):
        _results(show_vector=False)
        st.success(TAKEAWAYS[index])
        st.caption("Quan sát trên patch đang chọn; di chuyển center hoặc sửa pixel sẽ thay đổi kết quả.")


def guided_tutorial() -> None:
    _init()
    state = st.session_state
    step = int(state.sobel_guided_step)
    _header("Sobel Convolution · Guided Tutorial", "Từ chín pixel đến Gx, Gy và hướng thay đổi intensity.")
    with st.container(key="sobel_nav"):
        a, b, c = st.columns(3)
        a.button("← Previous", disabled=step == 1, on_click=_guided_move, args=(-1,), width="stretch")
        gate = (step == 6 and not state.sobel_reveal_gy) or (step == 12 and not state.sobel_reveal_final)
        b.button("Next →", disabled=step == 12 or gate, on_click=_guided_move, args=(1,), width="stretch")
        c.button("Reset", on_click=_guided_reset, width="stretch")
    _progress(step)
    st.subheader(f"Bước {step}/12 · {STEPS[step-1]}")

    if step == 1:
        _image(title="5×7 pixels · biên dọc giữa 20 và 200")
        st.caption("Mỗi ô là intensity 0–255. Chọn một ô để đọc giá trị; thay đổi đột ngột là dấu hiệu của local edge.")
        _pixel_inspect(edit=False)
        st.markdown('<div class="question">Vì sao thay đổi intensity đột ngột gợi ý một biên?</div>', unsafe_allow_html=True)
    elif step == 2:
        _image()
        _center_controls()
        result = compute_sobel_at(state.sobel_image, state.sobel_center)
        st.markdown("**Patch đúng tại center đã chọn**")
        st.markdown(_matrix_html(result.patch, color="#00a7bf"), unsafe_allow_html=True)
    elif step == 3:
        _image()
        _arithmetic("Gx", 0, compact=True)
        st.caption("Mask bên trái âm, bên phải dương: phép nhân đo thay đổi từ trái sang phải. Đây là cross-correlation, không lật mask.")
    elif step == 4:
        _image(active=divmod(state.sobel_progress, 3) if state.sobel_progress < 9 else None)
        _arithmetic("Gx", state.sobel_progress)
    elif step == 5:
        _image()
        _arithmetic("Gx", 9, compact=True)
        st.success("Ba hàng cộng lại tạo Gx; biên dọc sáng về bên phải cho Gx dương lớn.")
    elif step == 6:
        _image()
        st.radio("Dự đoán Gy của cùng patch", ("Lớn", "Nhỏ hoặc bằng 0"),
                 key="sobel_predict_gy", index=None, horizontal=True)
        st.button("Reveal Gy", disabled=state.get("sobel_predict_gy") is None,
                  on_click=lambda: state.update(sobel_reveal_gy=True))
        if state.sobel_reveal_gy:
            _arithmetic("Gy", 9, compact=True)
            st.caption("Cùng chín pixel, mask khác: so sánh phía trên và dưới. Với biên dọc mặc định, Gy=0.")
    elif step == 7:
        _image()
        _results()
        st.caption("Gradient là pháp tuyến cục bộ của biên thẳng lý tưởng; không phải edge tangent.")
    elif step == 8:
        _results()
        st.latex(r"M=\sqrt{G_x^2+G_y^2},\quad \theta=\operatorname{atan2}(G_y,G_x)")
        st.caption("atan2 giữ thông tin dấu/góc phần tư; với M=0, hướng undefined, không gán ý nghĩa cho atan2(0,0).")
    elif step == 9:
        _image()
        _center_controls()
        _results(show_vector=False)
        _map_view(default="Magnitude")
        st.caption("Mỗi ô interior được tính từ patch 3×3 riêng; border để trống vì không padding.")
    elif step == 10:
        _preset_controls()
        _image()
        _center_controls()
    elif step == 11:
        clean = create_sobel_presets()[PRESET_NAMES[0]]
        noisy = create_sobel_presets()[PRESET_NAMES[5]]
        st.radio("Dự đoán khi thêm noise", ("Gx/Gy không đổi", "Có thể đổi vì pixel đổi"),
                 key="sobel_predict_noise", index=None, horizontal=True)
        st.button("Reveal clean ↔ noisy", disabled=state.get("sobel_predict_noise") is None,
                  on_click=lambda: state.update(sobel_reveal_noise=True))
        st.plotly_chart(matrix_figure(clean, "Clean edge", center=DEFAULT_CENTER, patch=True), config=PLOT_CONFIG)
        if state.sobel_reveal_noise:
            st.plotly_chart(matrix_figure(noisy, "Fixed-seed noisy edge", center=DEFAULT_CENTER, patch=True), config=PLOT_CONFIG)
            a = compute_sobel_at(clean, DEFAULT_CENTER)
            b = compute_sobel_at(noisy, DEFAULT_CENTER)
            st.info(f"Clean: (Gx,Gy)=({a.gx},{a.gy}), M={a.magnitude:.1f}. "
                    f"Noisy: ({b.gx},{b.gy}), M={b.magnitude:.1f}. Noise cũng sinh gradient cục bộ.")
    else:
        _image()
        st.markdown('<div class="formula-box">Pixels → masks → 9 products → Gx, Gy → M, θ → gradient map</div>', unsafe_allow_html=True)
        st.radio("Sobel một mình bảo đảm edge map mảnh, sạch noise?", ("Có", "Không"),
                 key="sobel_predict_final", index=None, horizontal=True)
        st.button("Reveal kết luận", disabled=state.get("sobel_predict_final") is None,
                  on_click=lambda: state.update(sobel_reveal_final=True))
        if state.sobel_reveal_final:
            st.success("Không. Sobel đo thay đổi cục bộ; Canny thêm smoothing, non-maximum suppression và hysteresis. "
                       "Texture, noise, góc và mẫu phức tạp có thể tạo gradient không phải biên vật thể.")


def _pixel_inspect(*, edit=True) -> None:
    state = st.session_state
    h, w = state.sobel_image.shape
    a, b = st.columns(2)
    a.selectbox("Inspect row", range(h), key="sobel_pixel_row", on_change=_select_pixel)
    b.selectbox("Inspect column", range(w), key="sobel_pixel_col", on_change=_select_pixel)
    value = int(state.sobel_image[state.sobel_pixel_row, state.sobel_pixel_col])
    st.caption(f"Pixel [{state.sobel_pixel_row},{state.sobel_pixel_col}] = {value}")
    if edit:
        st.number_input("New intensity · 0–255", min_value=0, max_value=255,
                        step=1, key="sobel_edit_value")
        st.button("Apply pixel edit", on_click=_edit)
        if state.get("sobel_edit_error"):
            st.error(state.sobel_edit_error)


def _map_view(*, default="Gx") -> None:
    state = st.session_state
    layers = ("Gx", "Gy", "Magnitude", "Direction")
    st.radio("Output layer", layers, key="sobel_layer", horizontal=True,
             index=layers.index(default))
    maps = compute_gradient_maps(state.sobel_image)
    layer = state.sobel_layer
    with st.container(key="sobel_map_chart"):
        st.plotly_chart(matrix_figure(maps[layer], f"{layer} map · valid interior only",
                                      center=state.sobel_center, layer=layer,
                                      presentation=bool(state.get("sobel_presentation", False))), config=PLOT_CONFIG)
    row, col = state.sobel_center
    value = maps[layer][row, col]
    st.caption(f"Raw {layer}[{row},{col}] = {value:.2f} "
               "· màu chỉ là mapping để nhìn, không thay đổi phép tính. Border — = chưa tính.")


def experiment_lab() -> None:
    _init()
    state = st.session_state
    _header("Sobel Convolution · Experiment Lab", "Sửa pixel, di chuyển mask, chạy từng tích và khảo sát toàn bộ gradient map.")
    st.radio("Workspace", ("A · Pixel & Kernel Explorer", "B · Gradient Explorer"),
             key="sobel_workspace", horizontal=True)
    a, b = st.columns([2, 1])
    with a:
        st.selectbox("Teaching preset", PRESET_NAMES, key="sobel_preset", on_change=_load_preset)
    with b:
        st.button("Reset image to preset", on_click=_load_preset, width="stretch")
    index = PRESET_NAMES.index(state.sobel_preset)
    st.caption(f"Câu hỏi · {QUESTIONS[index]}")
    _image()
    _center_controls()
    if state.sobel_workspace.startswith("A"):
        st.subheader("Pixel & Kernel Explorer")
        _pixel_inspect()
        st.radio("Active mask", ("Gx", "Gy"), key="sobel_axis", horizontal=True, on_change=_set_axis)
        _arithmetic(state.sobel_axis, state.sobel_progress)
        _results(show_vector=False)
    else:
        st.subheader("Gradient Explorer")
        _map_view()
        _results()
        st.caption("Signed Gx/Gy có thang màu đối xứng; magnitude không âm. Direction dùng màu vòng theo góc. "
                   "Chỉ interior hợp lệ; M=0 có θ undefined.")
    with st.expander("Giải thích & giới hạn"):
        st.write(TAKEAWAYS[index])
        st.write("Sobel áp mask trực tiếp (cross-correlation), không lật như convolution toán học. "
                 "Noise, texture và góc có thể làm gradient lệch hướng; không có NMS/hysteresis ở đây.")
