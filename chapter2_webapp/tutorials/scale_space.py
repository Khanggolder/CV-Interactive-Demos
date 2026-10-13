"""Second Chapter 2 module: guided scale space → LoG → DoG → extrema."""

from __future__ import annotations

from html import escape

import cv2
import numpy as np
import streamlit as st

from utils.harris import pipeline as harris_pipeline, threshold_nms
from utils.scale_space import (PRESET_NAMES, PRESET_QUESTIONS, build_pyramid,
    detect_extrema, filter_contrast, gaussian_blur, gaussian_kernel,
    inspect_neighbors, log_dog_comparison, log_from_level, log_response, map_octave, presets,
    replay, scale_curve, semigroup)
from utils.scale_space_visualization import curve, heatmap, neighborhood_html

STEPS = (
    "Vì sao Harris cần nhiều thang đo?", "Không gian thang đo Gaussian", "Khám phá σ",
    "Tính chất nửa nhóm", "Dựng Gaussian Pyramid", "Octave và hai hệ σ",
    "Blob lớn, blob nhỏ", "Laplacian of Gaussian", "Chuẩn hoá theo σ²",
    "Difference of Gaussians", "Vì sao DoG xấp xỉ LoG?", "Kim tự tháp DoG",
    "Kiểm tra 26 láng giềng", "Thử đổi tỉ lệ", "Tổng kết",
)
PREDICTIONS = {
    1: ("Một cửa sổ Harris cố định có bất biến với resize?", ("Có", "Không"), "Không"),
    3: ("Chi tiết nào mờ đi trước khi tăng σ?", ("Nhỏ", "Lớn"), "Nhỏ"),
    7: ("Blob nhỏ hay lớn đạt đỉnh ở σ nhỏ hơn?", ("Blob nhỏ", "Blob lớn"), "Blob nhỏ"),
    8: ("LoG tại tâm blob sáng có dấu gì (quy ước Laplacian này)?", ("Âm", "Dương"), "Âm"),
    9: ("Có thể so biên độ LoG thô qua các σ mà không chuẩn hoá?", ("Có", "Không"), "Không"),
    11: ("DoG/(k−1) có bắt buộc bằng LoG chuẩn hoá từng pixel?", ("Có", "Không"), "Không"),
    15: ("Cực trị DoG thô đã là keypoint SIFT hoàn chỉnh?", ("Có", "Không"), "Không"),
}


def _init():
    s = st.session_state
    s.setdefault("scale_space_step", 1)
    s.setdefault("scale_space_mode", s.get("scale_space_saved_mode", "Guided Tutorial"))
    s.setdefault("scale_space_presentation", s.get("scale_space_saved_presentation", False))
    s.setdefault("scale_space_lab_a_preset", PRESET_NAMES[0])
    s.setdefault("scale_space_lab_b_preset", PRESET_NAMES[0])
    s.setdefault("scale_space_events_seen", 0)


def _save_view():
    st.session_state.scale_space_saved_mode = st.session_state.scale_space_mode
    st.session_state.scale_space_saved_presentation = st.session_state.scale_space_presentation


def _move(delta):
    st.session_state.scale_space_step = min(15, max(1, st.session_state.scale_space_step+delta))


def _reset(prefix="scale_space_"):
    for key in list(st.session_state):
        if key.startswith(prefix) and key not in ("scale_space_mode", "scale_space_presentation",
                                                   "scale_space_saved_mode", "scale_space_saved_presentation"):
            del st.session_state[key]
    st.session_state.scale_space_step = 1


def _plot(a, title, *, signed=False, point=None, markers=None, contrast_max=None):
    index = st.session_state["_scale_space_plot_index"]
    st.session_state["_scale_space_plot_index"] = index+1
    fig = heatmap(a, title, signed=signed, point=point, markers=markers,
                  presentation=bool(st.session_state.scale_space_presentation))
    if contrast_max is not None:
        fig.data[0].zmax = max(float(contrast_max), 1e-12)
        fig.data[0].colorbar.title = "|Δ| thực"
    st.plotly_chart(fig,
                    key=f"scale_space_plot_{index}", use_container_width=True,
                    config={"displayModeBar": False})


def _curve(x, series, title):
    index = st.session_state["_scale_space_plot_index"]
    st.session_state["_scale_space_plot_index"] = index+1
    st.plotly_chart(curve(x, series, title,
                          presentation=bool(st.session_state.scale_space_presentation)),
                    key=f"scale_space_plot_{index}", use_container_width=True,
                    config={"displayModeBar": False})


def _gate(step, explanation):
    question, choices, correct = PREDICTIONS[step]
    st.markdown(f'<div class="question">DỰ ĐOÁN · {escape(question)}</div>',
                unsafe_allow_html=True)
    choice = st.radio("Chọn trước khi reveal", choices, index=None,
                      key=f"scale_space_predict_{step}", horizontal=True)
    if st.button("Reveal kết quả", key=f"scale_space_reveal_button_{step}",
                 disabled=choice is None):
        st.session_state[f"scale_space_reveal_{step}"] = True
    revealed = bool(st.session_state.get(f"scale_space_reveal_{step}", False))
    if revealed:
        st.markdown(f'<div class="explain">{escape("Đúng" if choice == correct else "Chưa đúng")}. '
                    f'{escape(explanation)}</div>', unsafe_allow_html=True)
    return revealed


def _navigation(step, allowed):
    back, next_, reset = st.columns(3)
    back.button("← Previous", on_click=_move, args=(-1,), disabled=step == 1,
                width="stretch")
    next_.button("Next →", on_click=_move, args=(1,), disabled=step == 15 or not allowed,
                 width="stretch")
    reset.button("Reset", on_click=_reset, width="stretch")


def _pair(a, b, title_a, title_b, *, signed_a=False, signed_b=False, point=None):
    left, right = st.columns(2)
    with left:
        _plot(a, title_a, signed=signed_a, point=point)
    with right:
        _plot(b, title_b, signed=signed_b, point=point)


def _pyramid_view(pyramid, octave=0, layer=0):
    o = pyramid[octave]
    st.caption(f"Octave {octave} · tầng Gaussian j={layer} · "
               f"{o.gaussians[0].shape[1]}×{o.gaussians[0].shape[0]} px · "
               f"k = {2**(1/(len(o.gaussians)-3)):.6f}")
    st.write(f"σ_global = σ_local × 2^octave = {o.local_sigmas[layer]:.4f} × 2^{octave} "
             f"= {o.global_sigmas[layer]:.4f} px ảnh gốc")
    _pair(o.gaussians[layer], o.dogs[min(layer,len(o.dogs)-1)],
          f"L{octave},{layer} · σ_local={o.local_sigmas[layer]:.3f} · σ_global={o.global_sigmas[layer]:.3f}",
          f"D{octave},{min(layer,len(o.dogs)-1)} = L_next − L_current", signed_b=True)
    st.dataframe([{"octave": oc.index, "j": j, "kích thước": f"{g.shape[1]}×{g.shape[0]}",
                   "σ local": round(oc.local_sigmas[j],3),
                   "σ global": round(oc.global_sigmas[j],3),
                   "blur thêm": round(oc.increments[j],3)}
                  for oc in pyramid for j,g in enumerate(oc.gaussians)],
                 hide_index=True, width="stretch")


def _inspector(stack, layer, row, col, *, prefix, source_key):
    events = inspect_neighbors(stack, layer, row, col)
    signature = (source_key, layer, row, col)
    if st.session_state.get(prefix+"signature") != signature:
        st.session_state[prefix+"signature"] = signature
        st.session_state[prefix+"seen"] = 0
        st.session_state[prefix+"started"] = False
        st.session_state.pop(prefix+"prediction", None)
        st.session_state.pop(prefix+"revealed", None)
    center = float(stack[layer][row,col])
    st.write(f"Ứng viên D[{layer},{row},{col}] = {center:+.7f} · x={col}, y={row}")
    prediction = st.radio("Dự đoán kết quả 26 láng giềng", ("MAX", "MIN", "NOT EXTREMUM"),
                          index=None, key=prefix+"prediction", horizontal=True)
    started = bool(st.session_state.get(prefix+"started", False))
    with st.container(key="scale_space_inspector_controls"):
        a,b,c,d,e = st.columns(5)
        if a.button("Bắt đầu", key=prefix+"start", disabled=prediction is None):
            st.session_state[prefix+"started"] = True
            st.session_state[prefix+"seen"] = 0
            started = True
        seen = int(st.session_state.get(prefix+"seen", 0))
        if b.button("← So sánh", key=prefix+"prev", disabled=not started or seen == 0):
            seen -= 1
        if c.button("So sánh →", key=prefix+"next", disabled=not started or seen == 26):
            seen += 1
        if d.button("Chạy còn lại", key=prefix+"all", disabled=not started or seen == 26):
            seen = 26
        if e.button("Reset kiểm tra", key=prefix+"reset"):
            seen = 0
            st.session_state[prefix+"started"] = False
            st.session_state.pop(prefix+"revealed", None)
            started = False
    st.session_state[prefix+"seen"] = seen
    active = events[seen-1] if seen else None
    st.markdown(neighborhood_html(stack, layer, row, col, active), unsafe_allow_html=True)
    if active:
        st.write(f"So sánh {seen}/26: D[{active.layer},{active.row},{active.col}] = "
                 f"{active.value:+.7f}; tâm {center:+.7f} "
                 f"{'>' if center > active.value else '≤'} và "
                 f"{'<' if center < active.value else '≥'} láng giềng.")
        st.write(f"Trạng thái theo {seen} sự kiện: {replay(events, seen)}")
    else:
        st.caption("8 cùng tầng + 9 tầng thấp + 9 tầng cao; ô giữa không tự so với chính nó.")
    if st.button("Reveal quyết định", key=prefix+"reveal", disabled=prediction is None or seen < 26):
        st.session_state[prefix+"revealed"] = True
    revealed = bool(st.session_state.get(prefix+"revealed", False))
    if revealed:
        decision = replay(events, 26)
        actual = decision if decision in ("MAX", "MIN") else "NOT EXTREMUM"
        st.success(f"Kết luận strict: {actual}; dự đoán {'đúng' if prediction == actual else 'chưa đúng'}. "
                   "Giá trị bằng nhau bị loại; biên và tầng DoG ngoài cùng không hợp lệ.")
    return revealed


def _guided():
    step = int(st.session_state.scale_space_step)
    active = "".join(f'<span class="{"done" if i < step else "active" if i == step else ""}"></span>'
                     for i in range(1,16))
    st.markdown(f'<div class="chapter2-steps" aria-label="Bước {step}/15">{active}</div>',
                unsafe_allow_html=True)
    st.caption(f"BƯỚC {step}/15 · {STEPS[step-1]}")
    st.subheader(STEPS[step-1])
    images = presets()
    img = images[PRESET_NAMES[0]]
    blob = images[PRESET_NAMES[1]]
    pyramid = build_pyramid(img)
    allowed = True
    if step == 1:
        base = images[PRESET_NAMES[5]]
        scaled = cv2.resize(base, None, fx=.65, fy=.65, interpolation=cv2.INTER_AREA)
        _pair(base, scaled, "Góc gốc · 129×129", "Cùng góc resize · 84×84")
        st.caption("Resize thay đổi kích thước cấu trúc theo pixel; không tạo góc padding giả. "
                   "Cửa sổ tích phân Harris 5×5 vẫn giữ nguyên kích thước pixel.")
        allowed = _gate(1, "Một cửa sổ cố định có thể bỏ lỡ cấu trúc ở kích thước khác; "
                        "đa thang đo là động lực, không phải bảo đảm bất biến tuyệt đối.")
    elif step in (2,3):
        sigma = st.slider("σ · pixel", .5, 8.0, 1.6 if step == 2 else 3.0, .1,
                          key=f"scale_space_sigma_{step}")
        smooth = gaussian_blur(img, sigma)
        if step == 2:
            st.latex(r"L(x,y,\sigma)=G_\sigma*I,\qquad G_\sigma(x,y)=\frac{e^{-(x^2+y^2)/(2\sigma^2)}}{2\pi\sigma^2}")
        _pair(img, smooth, "Ảnh gốc", f"L(x,y,σ={sigma:.1f})")
        left,right = st.columns(2)
        with left:
            _plot(smooth[22:49,22:49], "Zoom 27×27 · blob nhỏ")
        with right:
            _plot(gaussian_kernel(sigma), f"Kernel Gaussian · bán kính ceil(4σ)={int(np.ceil(4*sigma))}")
        st.write(f"Tâm blob nhỏ: gốc {img[36,35]:.5f} → sau blur {smooth[36,35]:.5f}.")
        st.caption("Gaussian smoothing không resize và không tự nhận dạng feature. Kernel cắt ở ±4σ, "
                   "chuẩn hoá tổng 1; biên REFLECT_101.")
        if step == 3:
            allowed = _gate(3, "Chi tiết hẹp bị trung bình với nền sớm hơn; cấu trúc rộng tồn tại lâu hơn.")
    elif step == 4:
        sigma1 = st.slider("σ₁", .5, 3.0, 1.2, .1, key="scale_space_s1")
        sigma2 = st.slider("σ₂", sigma1+.1, 6.0, max(3.0,sigma1+.1), .1,
                           key="scale_space_s2")
        r = semigroup(img, sigma1, sigma2)
        _pair(r["direct"], r["incremental"], f"Trực tiếp σ₂={sigma2:.2f}",
              f"Tăng thêm σΔ={r['delta']:.3f}")
        _plot(r["difference"], "|Trực tiếp − tăng dần| · giá trị thực",
              contrast_max=r["max_error"])
        st.caption("Difference visualization contrast enhanced for inspection. "
                   "Color scale follows the measured maximum; hover values remain the original residual.")
        st.write(f"σΔ=√(σ₂²−σ₁²)={r['delta']:.5f}; MAE={r['mae']:.7f}, "
                 f"max={r['max_error']:.7f}. Dùng sai σ₂−σ₁: MAE={r['wrong_mae']:.7f}.")
        st.caption("Đẳng thức chính xác trong mô hình liên tục; kernel hữu hạn, sampling và biên gây lệch nhỏ.")
    elif step in (5,6,12):
        octave = st.selectbox("Octave", (0,1,2), key=f"scale_space_oct_{step}")
        layer = st.slider("Tầng Gaussian j", 0, 5, 0, key=f"scale_space_layer_{step}")
        _pyramid_view(pyramid, octave, layer)
        if step == 5:
            st.caption("Trong octave, chỉ blur thêm √(σ_target²−σ_previous²). "
                       "Sang octave sau: lấy mỗi pixel thứ hai từ tầng j=s đã blur chống alias.")
        elif step == 6:
            st.write(f"σ_local={pyramid[octave].local_sigmas[layer]:.4f}; "
                     f"σ_global={pyramid[octave].global_sigmas[layer]:.4f} px ảnh gốc. "
                     "x_global=2^o·x_local; y tương tự.")
        else:
            st.write("Mỗi octave có s+3=6 Gaussian, s+2=5 DoG; chỉ DoG j=1..3 "
                     "có hai tầng kề để xét cực trị. Các ảnh ở đây là mảng thật.")
    elif step in (7,8,9):
        sigma = st.slider("σ tại blob", .7, 12.0, 4.0, .1,
                          key=f"scale_space_blob_sigma_{step}")
        selected = st.selectbox("Blob", ("nhỏ r=4", "vừa r=9", "lớn r=16") if step == 7 else
                                ("sáng", "tối"), key=f"scale_space_blob_{step}")
        if step == 7:
            coords = {"nhỏ r=4":(36,35,4), "vừa r=9":(87,60,9), "lớn r=16":(56,95,16)}
            row,col,radius = coords[selected]
            target = img
        else:
            row=col=64
            radius=11
            target = images[PRESET_NAMES[1] if selected == "sáng" else PRESET_NAMES[2]]
        smooth, lap, normalized = log_response(target, sigma)
        sigmas = np.linspace(.7, 14, 50)
        raw, norm = scale_curve(target, row, col, sigmas)
        peak = int(np.argmax(np.abs(norm)))
        if step == 7:
            _pair(target, smooth, f"Blob · r={radius} px", f"Gaussian σ={sigma:.1f}")
            allowed = _gate(7, "Thang chọn bằng cực trị trị tuyệt đối của LoG chuẩn hoá, "
                            "không hardcode theo bán kính.")
        elif step == 8:
            _pair(smooth, lap, f"L(σ={sigma:.1f})", "LoG signed · Laplacian 5 điểm",
                  signed_b=True, point=(row,col))
            allowed = _gate(8, "Blob sáng có tâm cong xuống nên Laplacian âm; blob tối ngược dấu.")
        else:
            _pair(lap, normalized, "LoG thô", "σ²·LoG", signed_a=True, signed_b=True)
            allowed = _gate(9, "Đạo hàm bậc hai giảm biên độ theo thang; nhân σ² "
                            "cho phép so sánh scale response có nghĩa hơn.")
        if allowed:
            _curve(sigmas, {"LoG thô":raw, "σ²·LoG":norm}, "Đáp ứng tại tâm theo σ")
            st.write(f"Tâm (x={col},y={row}), r={radius} px: σ* đo được={sigmas[peak]:.3f} px; "
                     f"|σ²·LoG|={abs(norm[peak]):.6f}. r≈√2σ chỉ xấp xỉ theo mô hình blob.")
    elif step in (10,11):
        sigma = st.slider("σ thấp", .7, 6.0, 2.0, .1,
                          key=f"scale_space_dog_sigma_{step}")
        k = 2**(1/3)
        r = log_dog_comparison(blob, sigma, k)
        row = st.slider("y · hàng", 5, 123, 64, key=f"scale_space_dog_row_{step}")
        col = st.slider("x · cột", 5, 123, 64, key=f"scale_space_dog_col_{step}")
        if step == 10:
            _pair(r["low"], r["high"], f"L(σ={sigma:.3f})", f"L(kσ={sigma*k:.3f})",
                  point=(row,col))
            _plot(r["dog"], "DoG signed = L(kσ) − L(σ)", signed=True, point=(row,col))
            st.write(f"D[{row},{col}] = {r['high'][row,col]:.7f} − "
                     f"{r['low'][row,col]:.7f} = {r['dog'][row,col]:+.7f}.")
        else:
            allowed = _gate(11, "DoG/(k−1) xấp xỉ σ²·LoG khi k gần 1; "
                            "kernel hữu hạn và bước scale gây residual khác 0.")
            if allowed:
                _pair(r["normalized"], r["comparable"], "σ²·LoG", "DoG/(k−1)",
                      signed_a=True, signed_b=True)
                _plot(r["residual"], f"Residual signed · MAE={r['mae']:.6f}", signed=True)
                _curve(np.arange(blob.shape[1]),
                       {"σ²·LoG":r["normalized"][row],
                        "DoG/(k−1)":r["comparable"][row]}, "Profile hàng đã chọn")
                log_peak = np.unravel_index(np.argmax(np.abs(r["normalized"])), blob.shape)
                dog_peak = np.unravel_index(np.argmax(np.abs(r["comparable"])), blob.shape)
                st.write(f"Đỉnh |response| trên ảnh: LoG tại (x={log_peak[1]},y={log_peak[0]}); "
                         f"DoG/(k−1) tại (x={dog_peak[1]},y={dog_peak[0]}). "
                         f"MAE toàn ảnh={r['mae']:.6f}.")
                sampled = np.linspace(1.5, 10, 18)
                at_center = [log_dog_comparison(blob,float(s),k) for s in sampled]
                _curve(sampled,{"σ²·LoG tại tâm":[c["normalized"][64,64] for c in at_center],
                                "DoG/(k−1) tại tâm":[c["comparable"][64,64] for c in at_center]},
                       "Scale response · blob tâm")
                st.caption("Hai bản đồ không chuẩn hoá độc lập để giả vờ bằng nhau. "
                           "DoG tái dùng Gaussian đã tính nhưng pyramid vẫn có chi phí.")
    elif step == 13:
        o = pyramid[0]
        candidates = detect_extrema(o.dogs)
        options = [(c[0],c[1],c[2]) for c in candidates[:20]]
        options.append((2,64,64))
        layer,row,col = st.selectbox("Ứng viên (tầng, hàng, cột) · không tiết lộ loại",
                                      options, key="scale_space_candidate")
        st.write(f"Octave 0; 3×3×3 tại (x={col},y={row}); "
                 "chỉ kiểm tra tầng nội thất và pixel không ở biên.")
        allowed = _inspector(o.dogs, layer, row, col, prefix="scale_space_guided_inspect_",
                             source_key="guided-three-blobs-s3")
    elif step == 14:
        original = images[PRESET_NAMES[5]].copy()
        yy,xx = np.indices(original.shape)
        original[(xx-31)**2+(yy-34)**2 <= 8**2] = 1.0
        smaller = cv2.resize(original, (97,97), interpolation=cv2.INTER_AREA)
        big_h = harris_pipeline(original)
        small_h = harris_pipeline(smaller)
        h1 = threshold_nms(big_h.harris, .08*max(0,float(big_h.harris.max())))
        h2 = threshold_nms(small_h.harris, .08*max(0,float(small_h.harris.max())))
        left,right = st.columns(2)
        with left:
            _plot(original, f"Harris gốc · {int(h1.sum())} góc",
                  markers=[tuple(rc) for rc in np.argwhere(h1)])
        with right:
            _plot(smaller, f"Harris resize · {int(h2.sum())} góc",
                  markers=[tuple(rc) for rc in np.argwhere(h2)])
        d1 = build_pyramid(original, octaves=2)
        d2 = build_pyramid(smaller, octaves=2)
        c1 = [(o.index,*item) for o in d1 for item in filter_contrast(detect_extrema(o.dogs),.01)]
        c2 = [(o.index,*item) for o in d2 for item in filter_contrast(detect_extrema(o.dogs),.01)]
        def mapped(items, resize_factor=1):
            return [(int(round(r*2**o*resize_factor)),int(round(c*2**o*resize_factor)),
                     round(float(sigma*2**o*resize_factor),2))
                    for o,j,r,c,value,kind in items
                    for sigma in (d1[o].local_sigmas[j],)]
        mapped1 = mapped(c1)
        mapped2 = mapped(c2,129/97)
        st.write(f"DoG candidates sau |D|≥0.01: gốc {len(c1)}, resize {len(c2)}. "
                 "Tọa độ resize đã quy về hệ 129×129 bằng 129/97; "
                 "mỗi octave nhân 2^o. Không suy ra correspondence từ số lượng.")
        st.dataframe([{"nguồn":"gốc", "y quy gốc":r,"x quy gốc":c,"σ quy gốc":s}
                      for r,c,s in mapped1[:12]] +
                     [{"nguồn":"resize", "y quy gốc":r,"x quy gốc":c,"σ quy gốc":s}
                      for r,c,s in mapped2[:12]], hide_index=True, width="stretch")
        left,right = st.columns(2)
        with left:
            _plot(original,"DoG candidates · gốc",markers=[(r,c) for r,c,_ in mapped1])
        with right:
            _plot(smaller,"DoG candidates · resize",markers=[(r*97/129,c*97/129)
                                                                for r,c,_ in mapped2])
        st.caption("Ảnh không padding; pixel biên không đủ 3×3×3 bị loại. "
                   "Scale-space đặt nền cho SIFT, không bảo đảm matching hoặc invariance tuyệt đối.")
    else:
        st.write("Image → Gaussian smoothing → Scale Space → Gaussian Pyramid → "
                 "LoG/DoG → extrema (x,y,σ) → DoG extrema candidates.")
        allowed = _gate(15, "Cực trị DoG mới là ứng viên. Module SIFT kế tiếp sẽ refinement dưới pixel, "
                        "lọc edge response, gán orientation và tạo descriptor.")
    _navigation(step, allowed)


def _lab_a():
    st.subheader("Lab A · Gaussian Scale Space")
    controls = (st.expander("Điều khiển thí nghiệm", expanded=False)
                if st.session_state.scale_space_presentation else st.container())
    with controls:
        if st.button("Reset Lab A", key="scale_space_lab_a_reset"):
            for key in list(st.session_state):
                if key.startswith("scale_space_lab_a_"):
                    del st.session_state[key]
            st.rerun()
        c1,c2,c3,c4 = st.columns(4)
        preset = c1.selectbox("Preset", PRESET_NAMES, key="scale_space_lab_a_preset")
        sigma = c2.slider("σ", .5, 8.0, 2.0, .1, key="scale_space_lab_a_sigma")
        octaves = c3.slider("Octaves", 1, 3, 3, key="scale_space_lab_a_octaves")
        intervals = c4.slider("s · khoảng/octave", 2, 5, 3, key="scale_space_lab_a_intervals")
    image = presets()[preset]
    _pair(image, gaussian_blur(image,sigma), "Input", f"Gaussian σ={sigma:.1f}")
    st.caption(PRESET_QUESTIONS[PRESET_NAMES.index(preset)])
    pyramid = build_pyramid(image, octaves=octaves, intervals=intervals)
    o = st.selectbox("Xem octave", range(octaves), key="scale_space_lab_a_octave")
    j = st.slider("Gaussian layer", 0, intervals+2, 0, key="scale_space_lab_a_layer")
    _pyramid_view(pyramid,o,j)
    sigma1 = st.slider("Semigroup σ₁", .5, 2.5, 1.2, .1, key="scale_space_lab_a_s1")
    sigma2 = st.slider("Semigroup σ₂", 2.6, 6.0, 3.0, .1, key="scale_space_lab_a_s2")
    r = semigroup(image,sigma1,sigma2)
    st.write(f"σΔ={r['delta']:.4f}; MAE đúng={r['mae']:.7f}; "
             f"MAE σ₂−σ₁ sai={r['wrong_mae']:.7f}; max={r['max_error']:.7f}.")
    _plot(r["difference"], "|Direct − Incremental|")
    _plot(gaussian_kernel(sigma), f"Kernel hữu hạn ±ceil(4σ), σ={sigma:.1f}")


def _lab_b():
    st.subheader("Lab B · LoG, DoG & Extrema")
    with st.expander("Điều khiển · preset, octave, s, contrast", expanded=False):
        if st.button("Reset Lab B", key="scale_space_lab_b_reset"):
            for key in list(st.session_state):
                if key.startswith("scale_space_lab_b_") or key.startswith("scale_space_lab_inspect_"):
                    del st.session_state[key]
            st.rerun()
        c1,c2,c3,c4 = st.columns(4)
        preset = c1.selectbox("Preset", PRESET_NAMES, key="scale_space_lab_b_preset")
        intervals = c2.slider("s · khoảng/octave", 2, 5, 3, key="scale_space_lab_b_intervals")
        threshold = c3.slider("|DoG| tối thiểu · intensity [0,1]", 0.0, .1, .01, .001,
                              key="scale_space_lab_b_threshold")
        base_sigma = c4.slider("σ base · octave-local", 1.0, 2.5, 1.6, .1,
                               key="scale_space_lab_b_base_sigma")
        image = presets()[preset]
        pyramid = build_pyramid(image, intervals=intervals, base_sigma=base_sigma)
        octave = st.selectbox("Octave", (0,1,2), key="scale_space_lab_b_octave")
        o = pyramid[octave]
        layer = st.slider("DoG layer · chỉ 1..s có đủ láng giềng", 1, intervals, 1,
                          key="scale_space_lab_b_layer")
    candidates = detect_extrema(o.dogs)
    kept = filter_contrast(candidates, threshold)
    rejected = len(candidates)-len(kept)
    st.write(f"Raw strict 26-neighbor: {len(candidates)}; giữ |D|≥{threshold:.3f}: "
             f"{len(kept)}; loại contrast thấp: {rejected}. Đây chưa phải keypoint SIFT hoàn chỉnh.")
    raw = st.toggle("Hiện raw candidates", key="scale_space_lab_b_raw")
    filtered = st.toggle("Hiện candidates sau contrast", key="scale_space_lab_b_filtered")
    marked = [map_octave(r,c,0) for j,r,c,_,_ in (kept if filtered else candidates if raw else [])
              if j == layer]
    _pair(image, o.gaussians[layer], "Input", f"L · σ local={o.local_sigmas[layer]:.3f}, "
          f"global={o.global_sigmas[layer]:.3f}")
    _plot(o.dogs[layer], f"D{layer} · signed; {len(marked)} markers", signed=True, markers=marked)
    row = st.slider("y · octave-local", 1, o.dogs[layer].shape[0]-2,
                    o.dogs[layer].shape[0]//2, key="scale_space_lab_b_row")
    col = st.slider("x · octave-local", 1, o.dogs[layer].shape[1]-2,
                    o.dogs[layer].shape[1]//2, key="scale_space_lab_b_col")
    choices = [(j,r,c) for j,r,c,_,_ in candidates if j == layer]
    if choices:
        chosen = st.selectbox("Chọn ứng viên thật để kiểm tra · không hiện nhãn MAX/MIN",
                              ("Tọa độ thủ công",)+tuple(choices[:100]),
                              key="scale_space_lab_b_candidate")
        if chosen != "Tọa độ thủ công":
            _,row,col = chosen
    st.write(f"Global (x,y)=({map_octave(row,col,octave)[1]},"
             f"{map_octave(row,col,octave)[0]}); "
             f"D={o.gaussians[layer+1][row,col]:.7f}−"
             f"{o.gaussians[layer][row,col]:.7f}={o.dogs[layer][row,col]:+.7f}")
    lap, norm = log_from_level(o.gaussians[layer], o.local_sigmas[layer])
    _pair(lap,norm,f"LoG rời rạc · tầng j={layer}",
          f"σ_local²·LoG · σ={o.local_sigmas[layer]:.3f}",
          signed_a=True,signed_b=True)
    st.caption("LoG là Laplacian 5 điểm của Gaussian level đang chọn; nhân σ_local² của "
               "chính tầng đó. Gaussian level được xây tăng dần nên vẫn có sai khác rời rạc "
               "so với blur trực tiếp ảnh gốc.")
    _inspector(o.dogs,layer,row,col,prefix="scale_space_lab_inspect_",
               source_key=(preset,intervals,octave,base_sigma))
    sigmas = np.linspace(.7, min(14, max(image.shape)/8), 35)
    raw_curve,norm_curve = scale_curve(image,image.shape[0]//2,image.shape[1]//2,sigmas)
    _curve(sigmas,{"LoG":raw_curve,"σ²·LoG":norm_curve}, "Scale response tại tâm ảnh")
    scaled = cv2.resize(image, (97,97), interpolation=cv2.INTER_AREA)
    scaled_pyramid = build_pyramid(scaled,octaves=2,intervals=intervals,
                                   base_sigma=base_sigma)
    original_points = [(r*2**oc.index,c*2**oc.index) for oc in pyramid[:2]
                       for _,r,c,_,_ in filter_contrast(detect_extrema(oc.dogs),threshold)]
    scaled_points = [(r*2**oc.index,c*2**oc.index) for oc in scaled_pyramid
                     for _,r,c,_,_ in filter_contrast(detect_extrema(oc.dogs),threshold)]
    with st.expander("Đối chiếu ảnh gốc và resize · DoG candidates", expanded=False):
        left,right = st.columns(2)
        with left:
            _plot(image,f"Gốc 129×129 · {len(original_points)} ứng viên",
                  markers=original_points)
        with right:
            _plot(scaled,f"Resize 97×97 · {len(scaled_points)} ứng viên",
                  markers=scaled_points)
        st.caption("Ảnh resize không padding; tọa độ octave nhân 2^o, rồi tọa độ resize "
                   "nhân 129/97 để quy về ảnh gốc. Marker là candidate sau contrast; "
                   "số lượng và vị trí gần nhau chưa chứng minh correspondence hay full SIFT invariance.")


def render():
    _init()
    st.session_state["_scale_space_plot_index"] = 0
    presentation = bool(st.session_state.scale_space_presentation)
    st.markdown("""<style>
      @media(max-width:600px){.block-container{padding-top:1.2rem!important}
        header[data-testid="stHeader"]{height:0!important;visibility:hidden!important}
        .block-container h1{font-size:1.38rem!important;line-height:1.16;margin:.2rem 0!important}
        .block-container [data-testid="stVerticalBlock"]{gap:.45rem}
        .block-container [data-testid="stCaptionContainer"]{font-size:.8rem}
        .st-key-scale_space_inspector_controls [data-testid="stHorizontalBlock"]
          {flex-direction:row!important;flex-wrap:wrap!important;gap:5px!important}
        .st-key-scale_space_inspector_controls [data-testid="stColumn"]
          {min-width:calc(50% - 5px)!important;width:calc(50% - 5px)!important;flex:none!important}
        .st-key-scale_space_inspector_controls button{width:100%!important}}
      </style>""", unsafe_allow_html=True)
    if presentation:
        st.markdown("<style>.block-container{padding-top:.7rem!important}"
                    ".ss-cell{font-size:19px!important}</style>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="chapter-kicker">CHAPTER 2 · INTERACTIVE LEARNING LAB</div>',
                    unsafe_allow_html=True)
    st.title("Gaussian Scale Space · LoG · DoG")
    if not presentation:
        st.caption("Quan sát → dự đoán → thay đổi → tính thật → reveal → giải thích")
    left,right = st.columns([3,1])
    with left:
        st.radio("Chế độ", ("Guided Tutorial", "Experiment Lab"), horizontal=True,
                 key="scale_space_mode", on_change=_save_view)
    with right:
        st.toggle("Presentation Mode", key="scale_space_presentation", on_change=_save_view)
    if st.session_state.scale_space_mode == "Guided Tutorial":
        _guided()
    else:
        tab_a,tab_b = st.tabs(("Lab A · Gaussian", "Lab B · LoG/DoG"))
        with tab_a:
            _lab_a()
        with tab_b:
            _lab_b()
