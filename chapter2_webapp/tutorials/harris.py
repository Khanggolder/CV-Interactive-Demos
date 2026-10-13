"""Guided story and experiment lab; all numbers come from utils.harris."""

from __future__ import annotations

from html import escape

import numpy as np
import cv2
import streamlit as st

from utils.harris import (KX, KY, PRESET_NAMES, classify_eigenvalues, energy_grid,
                          integration_kernel, nms_decision, pipeline, presets,
                          shift_energy, shifted_patches, taylor_energy, tensor_at,
                          threshold_nms)
from utils.harris_visualization import eigenview, heatmap

STEPS = (
    "Tại sao cần local features?", "Phẳng · cạnh · góc", "Dịch cửa sổ",
    "Bản đồ E(u,v)", "Xấp xỉ Taylor", "Gradient Sobel", "Structure tensor",
    "Eigenvalue", "Harris R", "Response map", "Threshold và NMS",
    "Shi-Tomasi", "Giới hạn và biến đổi", "Tổng kết",
)
QUESTIONS = {
    2: ("Patch nào định vị được theo cả x và y?", ("Patch A", "Patch B", "Patch C")),
    4: ("Bản đồ E của một góc tăng theo hướng nào?", ("Chỉ theo x", "Chỉ theo y", "Cả x và y")),
    8: ("Khi cả λ₁ và λ₂ đều lớn, vùng ảnh là gì?", ("Vùng phẳng", "Cạnh", "Góc")),
    9: ("Harris R thường dương rõ tại đâu?", ("Vùng phẳng", "Cạnh", "Góc")),
    11: ("Hai đỉnh bằng nhau trong cùng lân cận NMS: giữ điểm nào?",
         ("Cả hai", "Điểm theo thứ tự hàng/cột nhỏ nhất", "Không điểm nào")),
    12: ("Shi-Tomasi dùng đại lượng nào?", ("max(λ₁,λ₂)", "min(λ₁,λ₂)", "trace(M)")),
    14: ("Harris một thang đo có tự bất biến với resize?", ("Có", "Không")),
}
ANSWERS = {2: "Patch C", 4: "Cả x và y", 8: "Góc", 9: "Góc",
           11: "Điểm theo thứ tự hàng/cột nhỏ nhất", 12: "min(λ₁,λ₂)", 14: "Không"}


def _init() -> None:
    state = st.session_state
    state.setdefault("chapter2_step", 1)
    state.setdefault("chapter2_mode", state.get("chapter2_saved_mode", "Guided Tutorial"))
    state.setdefault("chapter2_presentation", state.get("chapter2_saved_presentation", False))
    state.setdefault("chapter2_lab_preset", PRESET_NAMES[3])


def _save_view() -> None:
    st.session_state.chapter2_saved_mode = st.session_state.chapter2_mode
    st.session_state.chapter2_saved_presentation = st.session_state.chapter2_presentation


def _move(delta: int) -> None:
    st.session_state.chapter2_step = min(14, max(1, st.session_state.chapter2_step + delta))


def _reset() -> None:
    for key in list(st.session_state):
        if key.startswith("chapter2_predict_") or key.startswith("chapter2_reveal_"):
            del st.session_state[key]
    st.session_state.chapter2_step = 1


def _plot(values: np.ndarray, title: str, *, center=None, detections=None,
          colorscale="Viridis", displacement=False) -> None:
    fig = heatmap(values, title, center=center, detections=detections,
                  presentation=bool(st.session_state.chapter2_presentation),
                  colorscale=colorscale, displacement=displacement)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _gate(step: int, explanation: str) -> bool:
    question, choices = QUESTIONS[step]
    st.markdown(f'<div class="question">DỰ ĐOÁN · {escape(question)}</div>',
                unsafe_allow_html=True)
    choice = st.radio("Chọn trước khi xem kết quả", choices, index=None,
                      key=f"chapter2_predict_{step}", horizontal=True)
    if st.button("Reveal kết quả", key=f"chapter2_reveal_button_{step}", disabled=choice is None):
        st.session_state[f"chapter2_reveal_{step}"] = True
    revealed = bool(st.session_state.get(f"chapter2_reveal_{step}", False))
    if revealed:
        status = "Đúng" if choice == ANSWERS[step] else "Chưa đúng"
        st.markdown(f'<div class="explain">{status}. {escape(explanation)}</div>',
                    unsafe_allow_html=True)
    return revealed


def _navigation(step: int, allowed: bool) -> None:
    with st.container(key="chapter2_navigation"):
        back, forward, reset = st.columns(3)
        back.button("← Previous", on_click=_move, args=(-1,), disabled=step == 1,
                    use_container_width=True)
        forward.button("Next →", on_click=_move, args=(1,), disabled=step == 14 or not allowed,
                       use_container_width=True)
        reset.button("Reset", on_click=_reset, use_container_width=True)


def _guided() -> None:
    step = int(st.session_state.chapter2_step)
    active = "".join(f'<span class="{"done" if i < step else "active" if i == step else ""}"></span>'
                     for i in range(1, 15))
    st.markdown(f'<div class="chapter2-steps" aria-label="Bước {step}/14">{active}</div>',
                unsafe_allow_html=True)
    st.caption(f"BƯỚC {step}/14 · {STEPS[step-1]}")
    allowed = True
    examples = presets()
    base = examples[PRESET_NAMES[3]]
    center = (20, 20)
    result = pipeline(base)
    if step == 1:
        st.subheader("Điểm nào còn nhận ra khi ảnh thay đổi?")
        st.write("Ghép hai ảnh cùng cảnh cần những vị trí lặp lại, dễ định vị. Quan sát hai ảnh tổng hợp rồi tự hỏi: vùng nào sẽ giúp tìm lại vị trí theo cả hai chiều?")
        left, right = st.columns(2)
        with left:
            _plot(base, "Cảnh A · góc gốc", colorscale="Gray")
        with right:
            _plot(examples[PRESET_NAMES[6]], "Cảnh B · góc đã xoay", colorscale="Gray")
        st.caption("Minh họa tổng hợp cùng motif, chưa phải bài toán ghép ảnh đầy đủ.")
    elif step == 2:
        st.subheader("Ba cấu trúc, ba cách phản ứng khi dịch")
        cols = st.columns(3)
        for col, label, name in zip(cols, ("Patch A", "Patch B", "Patch C"),
                                     (PRESET_NAMES[0], PRESET_NAMES[1], PRESET_NAMES[3])):
            with col:
                _plot(examples[name][15:26, 15:26], label, colorscale="Gray")
        allowed = _gate(2, "Ở góc, dịch ngang và dọc đều đổi vùng nhìn thấy; cạnh chỉ xác định vị trí theo một hướng.")
        if allowed:
            st.caption("A = vùng phẳng · B = cạnh ngang · C = góc chữ L.")
    elif step == 3:
        st.subheader("Dịch thật một cửa sổ")
        choice = st.selectbox("Cấu trúc", PRESET_NAMES[:4], index=3, key="chapter2_shift_preset")
        image = examples[choice]
        c1, c2 = st.columns(2)
        u = c1.slider("u · dịch cột/x", -3, 3, 1, key="chapter2_guided_u")
        v = c2.slider("v · dịch hàng/y", -3, 3, 0, key="chapter2_guided_v")
        before, after = shifted_patches(image, *center, 9, u, v)
        a, b = st.columns(2)
        with a:
            _plot(before, "Cửa sổ gốc", colorscale="Gray")
        with b:
            _plot(after, "Cửa sổ sau dịch", colorscale="Gray")
        st.metric("E(u,v) = Σ [I(x+u,y+v)−I(x,y)]²", f"{shift_energy(image, *center, 9, u, v):.4f}")
        st.caption("Chỉ lấy vùng hợp lệ bên trong ảnh; không cuộn vòng, không padding. y tăng xuống dưới.")
    elif step == 4:
        st.subheader("Cùng một phép đo, nhiều hướng dịch")
        allowed = _gate(4, "Góc đổi cường độ theo hai hướng; vùng phẳng gần 0; cạnh ngang nhạy chủ yếu với v.")
        if allowed:
            names = (PRESET_NAMES[0], PRESET_NAMES[1], PRESET_NAMES[3])
            for col, name in zip(st.columns(3), names):
                with col:
                    _plot(energy_grid(examples[name], *center, 9), f"E(u,v) · {name}",
                          displacement=True)
            st.caption("Tâm (u=0,v=0) có E=0; v dương dịch xuống trong ảnh.")
    elif step == 5:
        st.subheader("Taylor: thay dịch ảnh bằng gradient")
        yy, xx = np.indices((41, 41))
        smooth = (xx + 2*yy).astype(float) / 120
        u = st.slider("u · dịch x", -2, 2, 1, key="chapter2_taylor_u")
        v = st.slider("v · dịch y", -2, 2, 1, key="chapter2_taylor_v")
        exact = shift_energy(smooth, *center, 7, u, v)
        approximate = taylor_energy(smooth, *center, 7, u, v)
        st.latex(r"E(u,v)=\sum_w[I(x+u,y+v)-I(x,y)]^2\;\approx\;\sum_w(uI_x+vI_y)^2")
        a, b = st.columns(2)
        a.metric("Dịch rời rạc chính xác", f"{exact:.6f}")
        b.metric("Xấp xỉ bậc nhất", f"{approximate:.6f}")
        st.caption("Fixture ramp tuyến tính, so trong vùng nội thất. Trên biên gắt hoặc dịch lớn, xấp xỉ không còn chính xác.")
    elif step == 6:
        st.subheader("Gradient: thay đổi theo x và y")
        st.write("Sobel là phép tương quan với kernel sau, chia 8 để ramp tăng 1/pixel cho đạo hàm xấp xỉ 1.")
        a, b = st.columns(2)
        with a:
            st.write("Kx =", KX)
            _plot(result.ix, "Iₓ · x sang phải", center=center)
        with b:
            st.write("Ky =", KY)
            _plot(result.iy, "Iᵧ · y xuống dưới", center=center)
        st.write(f"Tại (x=20,y=20): Iₓ={result.ix[center]:.4f}, Iᵧ={result.iy[center]:.4f}.")
    elif step == 7:
        st.subheader("Gộp gradient thành Structure Tensor")
        weighting = st.radio("Tích phân trong cửa sổ", ("Gaussian", "Box"), horizontal=True,
                             key="chapter2_guided_weighting")
        p = pipeline(base, weighting=weighting)
        matrix = tensor_at(p.maps, *center)
        st.latex(r"M=\begin{bmatrix}\langle I_x^2\rangle&\langle I_xI_y\rangle\\\langle I_xI_y\rangle&\langle I_y^2\rangle\end{bmatrix}")
        st.write(f"M tại (20,20), cửa sổ 5×5: [[{matrix[0,0]:.5f}, {matrix[0,1]:.5f}], "
                 f"[{matrix[1,0]:.5f}, {matrix[1,1]:.5f}]]")
        st.caption("Từng phần tử là tổng có trọng số của tích gradient; Gaussian σ=1.2 hoặc Box đều chuẩn hóa tổng trọng số = 1. Đây là thang tích phân, khác kernel Sobel 3×3 dùng lấy đạo hàm.")
        component = st.selectbox("Xem phép cộng tạo phần tử M", ("M₁₁ = ⟨Iₓ²⟩", "M₁₂ = ⟨IₓIᵧ⟩", "M₂₂ = ⟨Iᵧ²⟩"),
                                 key="chapter2_tensor_component")
        products = {"M₁₁ = ⟨Iₓ²⟩": p.ix*p.ix, "M₁₂ = ⟨IₓIᵧ⟩": p.ix*p.iy,
                    "M₂₂ = ⟨Iᵧ²⟩": p.iy*p.iy}
        weights = integration_kernel(5, 1.2, weighting)
        local = products[component][18:23, 18:23]
        st.write("Tích gradient 5×5 tại pixel:", np.round(local, 5))
        st.write("Trọng số tích phân 5×5:", np.round(weights, 5))
        st.write(f"Σ (tích × trọng số) = {np.sum(local*weights):.6f}.")
        _plot(p.maps[0], "Thành phần M₁₁ = ⟨Iₓ²⟩", center=center)
    elif step == 8:
        st.subheader("Hai eigenvalue nói điều gì?")
        allowed = _gate(8, "Cả hai hướng đều có biến thiên đủ mạnh, nên điểm được định vị trong 2D.")
        if allowed:
            name = st.selectbox("So sánh vùng", PRESET_NAMES[:4], index=3,
                                key="chapter2_eigen_preset")
            p = pipeline(examples[name])
            large, small = float(p.large[center]), float(p.small[center])
            st.plotly_chart(eigenview(large, small, st.session_state.chapter2_presentation),
                            use_container_width=True, config={"displayModeBar": False})
            st.write(f"λ₁={large:.5f}, λ₂={small:.5f} → {classify_eigenvalues(large, small)} "
                     "(mốc minh họa 0.005).")
            st.caption("Với E=c, bán trục theo eigenvector tỉ lệ 1/√λ. λ≈0 nghĩa là hướng không bị chặn; không vẽ elip hữu hạn giả.")
    elif step == 9:
        st.subheader("Harris dùng det và trace")
        allowed = _gate(9, "R dương đủ lớn khi hai eigenvalue đều lớn; cạnh có thể cho R âm. Dấu dương một mình chưa đủ: còn cần ngưỡng.")
        if allowed:
            k = st.slider("k", .03, .15, .05, .01, key="chapter2_guided_k")
            p = pipeline(base, k=k)
            m = tensor_at(p.maps, *center)
            st.latex(r"R=\det(M)-k\operatorname{trace}(M)^2")
            st.write(f"det={np.linalg.det(m):.6f}; trace={np.trace(m):.6f}; "
                     f"R={p.harris[center]:.6f} tại (20,20).")
            st.caption("R gần 0 thường là vùng phẳng, nhưng phân loại thực tế còn phụ thuộc độ lớn và ngưỡng.")
    elif step == 10:
        st.subheader("R là một bản đồ, không phải danh sách góc viết sẵn")
        k = st.slider("k", .03, .15, .05, .01, key="chapter2_map_k")
        p = pipeline(base, k=k)
        row = st.slider("y = hàng", 3, 37, 20, key="chapter2_map_row")
        col = st.slider("x = cột", 3, 37, 20, key="chapter2_map_col")
        _plot(p.harris, "Harris response", center=(row, col))
        st.write(f"R(x={col}, y={row}) = {p.harris[row,col]:.6f}.")
    elif step == 11:
        st.subheader("Threshold → cực đại cục bộ → góc")
        allowed = _gate(11, "NMS xét giá trị lớn hơn ngưỡng; nếu đồng hạng trong cùng cửa sổ, điểm có (hàng,cột) nhỏ nhất thắng. Bỏ dải biên nửa cửa sổ.")
        if allowed:
            p = pipeline(base)
            fraction = st.slider("Ngưỡng · tỉ lệ đỉnh R dương", 0.0, 0.5, 0.08, .01,
                                 key="chapter2_nms_threshold")
            threshold = fraction * max(0, float(p.harris.max()))
            mask = threshold_nms(p.harris, threshold, 5)
            candidates = [tuple(map(int, rc)) for rc in np.argwhere(p.harris > threshold)]
            candidates.sort(key=lambda rc: (-p.harris[rc], rc[0], rc[1]))
            if candidates:
                index = st.slider("Xem từng ứng viên · theo R giảm dần", 1, len(candidates), 1,
                                  key="chapter2_nms_candidate")
                row, col = candidates[index-1]
                keep, reason = nms_decision(p.harris, threshold, 5, row, col)
                overview, detail = st.columns([3, 2])
                with overview:
                    _plot(p.harris, "Toàn ảnh · dấu + là ứng viên đang xét",
                          center=(row, col), detections=mask)
                with detail:
                    if 2 <= row < 39 and 2 <= col < 39:
                        _plot(p.harris[row-2:row+3, col-2:col+3],
                              "Lân cận NMS 5×5 · tâm đang xét", center=(2, 2))
                    else:
                        st.info("Ứng viên ở dải biên: không có cửa sổ NMS 5×5 hợp lệ.")
                st.write(f"Ứng viên {index}/{len(candidates)} · (x={col},y={row}): "
                         f"R={p.harris[row,col]:.6f}; ngưỡng={threshold:.6f} → {reason}.")
                st.caption(f"Quyết định NMS: {'KEEP' if keep else 'REJECT'} · so R ở tâm với đúng cửa sổ và ngưỡng bên trên.")
            else:
                _plot(p.harris, "Không có ứng viên vượt ngưỡng", detections=mask)
    elif step == 12:
        st.subheader("Shi-Tomasi: tiêu chí nào cho góc tốt?")
        allowed = _gate(12, "Nếu λ₂ còn lớn thì thay đổi theo cả hai hướng; cùng tensor M, chỉ khác hàm response.")
        if allowed:
            p = pipeline(base)
            fraction = st.slider("Ngưỡng · tỉ lệ đỉnh dương", 0.0, 0.5, .08, .01,
                                 key="chapter2_shi_threshold")
            a, b = st.columns(2)
            with a:
                h_threshold = fraction * max(0, float(p.harris.max()))
                _plot(p.harris, "Harris · R", detections=threshold_nms(p.harris, h_threshold, 5))
            with b:
                s_threshold = fraction * max(0, float(p.shi_tomasi.max()))
                _plot(p.shi_tomasi, "Shi-Tomasi · min(λ)",
                      detections=threshold_nms(p.shi_tomasi, s_threshold, 5))
            st.caption("Ngưỡng tương đối áp dụng riêng cho mỗi response; giá trị thô không cùng đơn vị.")
    elif step == 13:
        st.subheader("Xoay, đổi tỉ lệ và nhiễu")
        choice = st.selectbox("Thử biến đổi", PRESET_NAMES[5:], key="chapter2_robust_preset")
        image = examples[choice]
        p = pipeline(image)
        threshold = .08 * max(0, float(p.harris.max()))
        mask = threshold_nms(p.harris, threshold)
        if choice in (PRESET_NAMES[6], PRESET_NAMES[7]):
            angle, scale = (28, 1.0) if choice == PRESET_NAMES[6] else (0, .6)
            middle = (base.shape[0]-1)/2
            transform = cv2.getRotationMatrix2D((middle,middle),angle,scale)
            support = cv2.warpAffine(np.ones_like(base),transform,
                                     (base.shape[1],base.shape[0]),
                                     flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0) > .999
            interior = cv2.erode(support.astype(np.uint8),np.ones((5,5),np.uint8)) > 0
            display_mask = mask & interior
        else:
            display_mask = mask
        base_threshold = .08 * max(0, float(result.harris.max()))
        base_mask = threshold_nms(result.harris, base_threshold)
        a, b = st.columns(2)
        with a:
            _plot(base, f"Gốc · {int(base_mask.sum())} điểm", detections=base_mask,
                  colorscale="Gray")
        with b:
            _plot(image, f"{choice} · {int(display_mask.sum())} điểm trong support", detections=display_mask,
                  colorscale="Gray")
        if choice in (PRESET_NAMES[6],PRESET_NAMES[7]):
            st.caption(f"Warp dùng INTER_LINEAR, BORDER_CONSTANT=0. Detector tìm {int(mask.sum())} đỉnh; "
                       f"chỉ đánh dấu {int(display_mask.sum())} đỉnh có cửa sổ 5×5 nằm hoàn toàn trong "
                       "vùng ảnh nguồn được hỗ trợ. Đỉnh gần biên đệm không là bằng chứng về độ ổn định theo scale.")
        st.write("Response tensor có tính bất biến quay trong lý thuyết liên tục, nhưng nội suy và xử lý biên làm điểm phát hiện rời rạc thay đổi. Detector một thang đo không tự bất biến với đổi tỉ lệ. Nhiễu có thể tạo đỉnh giả.")
    else:
        st.subheader("Ghép lại toàn bộ pipeline")
        st.write("Input → Sobel gradient → tích gradient → làm mịn tensor → Harris hoặc Shi-Tomasi → threshold + NMS → corners.")
        allowed = _gate(14, "Resize đổi kích thước cấu trúc so với cửa sổ tích phân cố định; cần cách xử lý đa thang đo ở bài sau.")
        if allowed:
            st.write("Tự kiểm tra thêm: vì sao cạnh không định vị được dọc theo chính nó? Gaussian ở tensor khác gì Sobel? Nếu tăng ngưỡng, điểm nào bị loại trước?")
    _navigation(step, allowed)


def _lab() -> None:
    presentation = bool(st.session_state.chapter2_presentation)
    st.subheader("Experiment Lab")
    preset_col, reset_col = st.columns([4, 1])
    with reset_col:
        reset = st.button("Reset Lab", key="chapter2_lab_reset", use_container_width=True)
    if reset:
        for key in list(st.session_state):
            if key.startswith("chapter2_lab_") and key != "chapter2_lab_reset":
                del st.session_state[key]
        st.rerun()
    with preset_col:
        preset = st.selectbox("Preset", PRESET_NAMES, index=3, key="chapter2_lab_preset")
    image = presets()[preset]
    with st.expander("Cửa sổ và dịch chuyển", expanded=False):
        a, b, c = st.columns(3)
        row = a.slider("y · hàng", 6, 34, 20, key="chapter2_lab_row")
        col = b.slider("x · cột", 6, 34, 20, key="chapter2_lab_col")
        size = c.select_slider("Cửa sổ", (3, 5, 7, 9), value=5, key="chapter2_lab_size")
        u = a.slider("u · dịch x", -2, 2, 1, key="chapter2_lab_u")
        v = b.slider("v · dịch y", -2, 2, 0, key="chapter2_lab_v")
        st.write(f"E chính xác = {shift_energy(image, row, col, size, u, v):.5f}; "
                 f"Taylor ≈ {taylor_energy(image, row, col, size, u, v):.5f}.")
    with st.expander("Tensor, response và NMS · mở để điều chỉnh", expanded=False):
        a, b, c = st.columns(3)
        weighting = a.selectbox("Tích phân", ("Gaussian", "Box"), key="chapter2_lab_weighting")
        sigma = b.slider("σ tích phân", .5, 3.0, 1.2, .1, key="chapter2_lab_sigma",
                         disabled=weighting == "Box")
        k = c.slider("Harris k", .03, .15, .05, .01, key="chapter2_lab_k")
        nms_size = a.select_slider("NMS window", (3, 5, 7), value=5, key="chapter2_lab_nms")
        fraction = b.slider("Ngưỡng · phần đỉnh dương", 0.0, .5, .08, .01,
                            key="chapter2_lab_threshold")
        algorithm = c.radio("Response", ("Harris", "Shi-Tomasi"), horizontal=True,
                            key="chapter2_lab_algorithm")
    p = pipeline(image, size, sigma, weighting, k)
    response = p.harris if algorithm == "Harris" else p.shi_tomasi
    absolute = fraction * max(0, float(response.max()))
    mask = threshold_nms(response, absolute, nms_size)
    m = tensor_at(p.maps, row, col)
    left, right = st.columns(2)
    with left:
        _plot(image, f"{preset} · pixel ({col},{row})", center=(row, col), colorscale="Gray")
    with right:
        _plot(response, f"{algorithm} · {int(mask.sum())} điểm giữ lại", center=(row, col),
              detections=mask)
    st.write(f"Iₓ={p.ix[row,col]:.5f}, Iᵧ={p.iy[row,col]:.5f}; "
             f"M=[[{m[0,0]:.5f},{m[0,1]:.5f}],[{m[1,0]:.5f},{m[1,1]:.5f}]]")
    st.write(f"λ₁={p.large[row,col]:.5f}, λ₂={p.small[row,col]:.5f}; "
             f"Harris R={p.harris[row,col]:.6f}; Shi-Tomasi={p.shi_tomasi[row,col]:.5f}; "
             f"ngưỡng {algorithm}={absolute:.6f}.")
    layer = st.selectbox("Xem lớp trung gian", ("Iₓ", "Iᵧ", "M₁₁", "M₁₂", "M₂₂", "Harris", "Shi-Tomasi"),
                         key="chapter2_lab_layer")
    layers = {"Iₓ": p.ix, "Iᵧ": p.iy, "M₁₁": p.maps[0], "M₁₂": p.maps[1],
              "M₂₂": p.maps[2], "Harris": p.harris, "Shi-Tomasi": p.shi_tomasi}
    _plot(layers[layer], f"Lớp {layer}", center=(row, col))
    st.caption("NMS: giá trị phải > ngưỡng tuyệt đối; bỏ biên bán kính NMS; tie giữ tọa độ (hàng,cột) nhỏ nhất trong lân cận.")


def render() -> None:
    _init()
    presentation = bool(st.session_state.chapter2_presentation)
    if presentation:
        st.markdown("<style>.block-container{padding-top:.8rem!important}</style>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="chapter-kicker">CHAPTER 2 · INTERACTIVE LEARNING LAB</div>',
                    unsafe_allow_html=True)
    st.title("Harris & Shi-Tomasi" if presentation else "Vì sao Harris tìm được góc?")
    if not presentation:
        st.caption("Từ dịch cửa sổ thật đến eigenvalue, Harris R và Shi-Tomasi · mọi số liệu tính trực tiếp")
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.radio("Chế độ", ("Guided Tutorial", "Experiment Lab"), horizontal=True,
                 key="chapter2_mode", on_change=_save_view)
    with top_right:
        st.toggle("Presentation Mode", key="chapter2_presentation", on_change=_save_view)
    if st.session_state.chapter2_mode == "Guided Tutorial":
        _guided()
    else:
        _lab()
