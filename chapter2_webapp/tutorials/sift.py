"""SIFT teaching module: computed evidence with staged prediction and three labs."""

from __future__ import annotations

from collections import Counter
from html import escape

import numpy as np
import streamlit as st

from utils.sift import (apply_lowe_ratio_test, build_sift_descriptor,
    calculate_dog_derivatives, calculate_orientation_histogram,
    calculate_spatial_hessian, descriptor_index, edge_decision,
    evaluate_geometric_correspondence, match_descriptors, pipeline,
    sample_oriented_descriptor_patch, teaching_presets, transform_scene)
from utils.sift_visualization import (descriptor_grid, eight_bins, gradient_arrows,
    image_view, matching_view, orientation_histogram, oriented_patch_view,
    response_patch, vector_view)

STEPS = (
    "SIFT giải quyết điều gì?", "Nhắc lại ứng viên DoG", "Cực trị trên lưới và subpixel",
    "Gradient và Hessian Taylor", "Giải độ lệch δ", "Loại tương phản thấp",
    "Loại đáp ứng dạng cạnh", "Dòng lọc keypoint", "Vì sao cần gán hướng?",
    "Histogram 36 hướng", "Hướng chủ đạo và hướng phụ", "Khung tọa độ xoay",
    "Lưới descriptor 4×4", "Ghép 128 thành phần", "Chuẩn hóa L2 → clip → L2",
    "Tìm hai descriptor gần nhất", "Lowe ratio test", "Toàn bộ pipeline và giới hạn",
)
GATES = {
    3:("Cực trị rời rạc nhất thiết trùng vị trí cực trị Taylor?",("Có","Không"),"Không"),
    5:("Nếu |δx| ≥ 0.5, ta có thể giữ nguyên ô lưới?",("Có","Không"),"Không"),
    6:("|D̂| thấp hơn ngưỡng sẽ được giữ?",("Có","Không"),"Không"),
    7:("Hessian cạnh ở đây có phải structure tensor Harris?",("Có","Không"),"Không"),
    10:("Góc gradient được tính từ atan2(Ly,Lx)?",("Có","Không"),"Có"),
    14:("Một ô có tám bin thì 16 ô tạo 128 chiều?",("Có","Không"),"Có"),
    16:("Gần nhất về descriptor có chắc đúng hình học?",("Có","Không"),"Không"),
    17:("Ratio bằng đúng ngưỡng 0.8 sẽ được chấp nhận?",("Có","Không"),"Không"),
    18:("SIFT đảm bảo bất biến tuyệt đối với mọi phép quay/tỉ lệ?",("Có","Không"),"Không"),
}
PRESET_GOAL = {
    "Blob và góc rõ":"Quan sát nhiều cực trị có hình dạng và hướng khác nhau.",
    "Blob đa thang":"Blob có kích thước khác nhau xuất hiện ở các octave/sigma khác nhau.",
    "Cấu trúc bất đối xứng":"Một ứng viên dạng cạnh có thể bị loại bởi Hessian DoG.",
    "Tương phản yếu":"Tăng ngưỡng để loại đáp ứng nhỏ; không hardcode số điểm.",
    "Cạnh kéo dài":"Biên có độ cong không cân đối hoặc không đủ cực trị strict.",
    "Hoa văn lặp":"Nhiều descriptor tương tự gây match mơ hồ.",
    "Cảnh biến đổi affine":"Đánh giá match bằng transform đã biết, chỉ sau khi matching.",
    "Cảnh nhiễu":"Nhiễu tạo ứng viên yếu; ngưỡng và lọc ảnh hưởng kết quả.",
}


@st.cache_data(show_spinner=False, max_entries=32)
def _compute(name, intervals=3, sigma=1.6, contrast=.008, edge=10., describe=True):
    image = teaching_presets()[name]
    return (image,*pipeline(image,intervals=intervals,base_sigma=sigma,
                            contrast=contrast,edge_ratio=edge,with_descriptors=describe))


@st.cache_data(show_spinner=False, max_entries=12)
def _matching(name, angle, scale, brightness, gain, noise):
    image = teaching_presets()[name]
    other,matrix = transform_scene(image,angle,scale,brightness,gain,noise)
    pa,ra,fa,da = pipeline(image)
    pb,rb,fb,db = pipeline(other)
    return image,other,matrix,da,db


@st.cache_data(show_spinner=False, max_entries=12)
def _rotation_compare(name,angle):
    rotated,matrix=transform_scene(teaching_presets()[name],angle)
    _,_,_,points=pipeline(rotated)
    return rotated,matrix,points


def _init():
    s=st.session_state
    s.setdefault("sift_step",1)
    s.setdefault("sift_mode",s.get("sift_saved_mode","Guided Tutorial"))
    s.setdefault("sift_presentation",s.get("sift_saved_presentation",False))


def _save_view():
    st.session_state.sift_saved_mode=st.session_state.sift_mode
    st.session_state.sift_saved_presentation=st.session_state.sift_presentation


def _reset(prefix="sift_"):
    keep={"sift_mode","sift_presentation","sift_saved_mode","sift_saved_presentation"}
    for key in list(st.session_state):
        if key.startswith(prefix) and key not in keep:
            del st.session_state[key]
    st.session_state.sift_step=1


def _move(delta):
    st.session_state.sift_step=min(18,max(1,st.session_state.sift_step+delta))


def _plot(fig):
    idx=st.session_state.get("sift_plot_count",0)
    st.session_state.sift_plot_count=idx+1
    st.plotly_chart(fig,key=f"sift_plot_{idx}",use_container_width=True,
                    config={"displayModeBar":False})


def _presentation():
    return bool(st.session_state.sift_presentation)


def _matrix(values,labels):
    """Compact projector-readable numeric matrix rather than a full-width dataframe."""
    rows=[]
    for label,row in zip(labels,np.asarray(values)):
        cells="".join(f"<td>{float(value):+.6f}</td>" for value in row)
        rows.append(f"<tr><th>{escape(label)}</th>{cells}</tr>")
    st.markdown('<style>.sift-matrix{border-collapse:collapse;margin:.6rem 0 1rem;'
                'font-variant-numeric:tabular-nums;font-size:1.05rem;max-width:100%;}'
                '.sift-matrix td,.sift-matrix th{border:1px solid #bacbd8;'
                'padding:.45rem .65rem;text-align:right;color:#172033;}'
                '.sift-matrix th{background:#eaf1f5;color:#123b5d;}'
                '@media(max-width:600px){.sift-matrix{font-size:.78rem;}'
                '.sift-matrix td,.sift-matrix th{padding:.28rem .34rem;}}</style>'
                f'<table class="sift-matrix">{"".join(rows)}</table>',unsafe_allow_html=True)


def _gate(step, explanation, signature=None):
    if signature is not None and st.session_state.get(f"sift_signature_{step}") != signature:
        st.session_state[f"sift_signature_{step}"]=signature
        st.session_state.pop(f"sift_predict_{step}",None)
        st.session_state.pop(f"sift_reveal_{step}",None)
    question,options,answer=GATES[step]
    st.markdown(f'<div class="question">DỰ ĐOÁN · {question}</div>',unsafe_allow_html=True)
    choice=st.radio("Chọn trước khi reveal",options,index=None,key=f"sift_predict_{step}",horizontal=True)
    if st.button("Reveal kết quả",key=f"sift_reveal_button_{step}",disabled=choice is None):
        st.session_state[f"sift_reveal_{step}"]=True
    revealed=bool(st.session_state.get(f"sift_reveal_{step}",False))
    if revealed:
        st.success(("Đúng. " if choice==answer else "Chưa đúng. ")+explanation)
    return revealed


def _nav(step,allowed):
    a,b,c=st.columns(3)
    a.button("← Previous",on_click=_move,args=(-1,),disabled=step==1,width="stretch")
    b.button("Next →",on_click=_move,args=(1,),disabled=step==18 or not allowed,width="stretch")
    c.button("Reset",on_click=_reset,width="stretch")


def _select_candidate(raw,filtered,key,preferred="kept",show_status=True):
    if not raw:
        st.warning("Không có cực trị strict trong preset/tham số này. Thử một ảnh khác.")
        return None,None
    indexes=list(range(len(raw)))
    if key not in st.session_state:
        st.session_state[key]=next((i for i,p in enumerate(filtered) if p.status==preferred),0)
    index=st.selectbox("Ứng viên DoG",indexes,key=key,
                       format_func=lambda i:f"#{i} · o={raw[i].octave}, D={raw[i].layer}, "
                       f"(x={raw[i].col},y={raw[i].row})"+
                       (f" · {filtered[i].status}" if show_status else ""))
    return raw[index],filtered[index]


def _counts(filtered,raw):
    counts=Counter(p.status for p in filtered)
    labels=("Raw","Không hội tụ","Contrast","Edge","Giữ lại")
    values=(len(raw),counts["invalid"],counts["low contrast"],counts["edge"],counts["kept"])
    cards="".join(f'<div class="sift-count"><small>{escape(label)}</small><strong>{value}</strong></div>'
                  for label,value in zip(labels,values))
    st.markdown('<style>.sift-counts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));'
                'gap:9px;margin:.5rem 0;}.sift-count{padding:.55rem .7rem;border:1px solid #d3e1e8;'
                'border-radius:9px;background:#fff;color:#172033}.sift-count small{display:block;'
                'font-size:.8rem;color:#3d4b60}.sift-count strong{display:block;font-size:1.5rem;'
                'line-height:1.2}@media(max-width:600px){.sift-counts{grid-template-columns:'
                'repeat(2,minmax(0,1fr));gap:6px}.sift-count{padding:.4rem .55rem}'
                '.sift-count strong{font-size:1.25rem}}</style>'
                f'<div class="sift-counts">{cards}</div>',unsafe_allow_html=True)
    st.caption("Mỗi ứng viên chỉ ở một nhóm: refinement fail → contrast fail → edge fail → kept.")


def _taylor(pyramid,raw,kp):
    t=calculate_dog_derivatives(pyramid[raw.octave].dogs,raw.layer,raw.row,raw.col)
    _plot(response_patch(pyramid[raw.octave].dogs,raw,presentation=_presentation()))
    st.write(f"g = (Dx,Dy,Ds) = {np.array2string(t.gradient,precision=7)}")
    _matrix(t.hessian,("x","y","s"))
    st.caption("Hessian 3×3 theo x=cột, y=hàng, s=tầng DoG; sai phân tâm bước 1, đạo hàm hỗn hợp /4.")
    if kp.status != "invalid":
        st.write(f"δ = ({kp.delta[0]:+.4f}, {kp.delta[1]:+.4f}, {kp.delta[2]:+.4f}); "
                 f"(x̂,ŷ,ŝ)=({kp.col_local:.4f},{kp.row_local:.4f},{kp.layer_local:.4f}); "
                 f"D̂={kp.response:+.7f}.")
        st.caption(f"σ local={kp.sigma_local:.3f}; σ global={kp.sigma_global:.3f}. "
                   "ŝ là chỉ số tầng phân số; σ được suy ra từ lịch Gaussian, không phải ŝ pixel.")
    else:
        st.warning(kp.reason)


def _orientation(pyramid,kp,*,show_sample=False,sample_key="sift_guided_orientation_sample"):
    info=calculate_orientation_histogram(kp,pyramid[kp.octave])
    selected=None
    if show_sample and info.contributions:
        selected=st.slider("Pixel gradient trong support",0,len(info.contributions)-1,
                           min(len(info.contributions)//2,len(info.contributions)-1),
                           key=sample_key)
    _plot(orientation_histogram(info,selected,presentation=_presentation()))
    with st.expander("Xem vector gradient thật trên Gaussian patch",expanded=_presentation()):
        _plot(gradient_arrows(pyramid[kp.octave],kp,info,selected,presentation=_presentation()))
    if selected is not None:
        row,col,mag,angle,spatial,weight=info.contributions[selected]
        st.write(f"Pixel (x={col},y={row}): |∇L|={mag:.5f}, θ={angle:.1f}°, "
                 f"Gaussian={spatial:.5f}, trọng số={weight:.5f} → hai bin lân cận.")
    st.write(f"Tầng Gaussian gần nhất: {info.gaussian_layer}; hướng đỉnh: "
             +(", ".join(f"{p:.0f}°" for p in info.peaks) if info.peaks else "không xác định"))
    st.caption("36 bin, nội suy hai bin và làm mượt vòng 5 điểm. Các cực đại ≥80% đỉnh "
               "được tạo thành orientation records thật; góc dương theo trục y ảnh hướng xuống.")
    return info


def _descriptor(pyramid,oriented,*,prefix):
    desc=build_sift_descriptor(oriented,pyramid[oriented.octave])
    patch,valid=sample_oriented_descriptor_patch(oriented,pyramid[oriented.octave])
    st.caption(f"(x,y)=({oriented.col_global:.2f},{oriented.row_global:.2f}) ảnh gốc · "
               f"σ global={oriented.sigma_global:.2f} · θ*={oriented.orientation:.0f}°; "
               f"support thực phụ thuộc σ, {int(valid.sum())}/{valid.size} mẫu patch trong ảnh.")
    left,right=st.columns(2)
    with left:
        _plot(oriented_patch_view(patch,presentation=_presentation()))
    row=st.selectbox("Hàng ô",range(4),key=prefix+"cell_row")
    col=st.selectbox("Cột ô",range(4),key=prefix+"cell_col")
    with right:
        _plot(descriptor_grid(desc.cells,(row,col),presentation=_presentation()))
    _plot(eight_bins(desc.cells[row,col],f"Ô ({row},{col}) · chỉ số {descriptor_index(row,col,0)}–{descriptor_index(row,col,7)}",
                     presentation=_presentation()))
    st.caption("Row-major: index=(cell_row·4+cell_col)·8+orientation_bin. "
               "Mỗi gradient được nội suy trilinear theo 2 ô không gian và 2 bin hướng; bỏ mẫu ngoài ảnh.")
    return desc


def _matching_panel(data,threshold=.8,prefix="sift_guided_match"):
    a,b,matrix,pa,pb=data
    matches=match_descriptors(pa,pb,threshold)
    if not matches:
        st.warning("Chưa có descriptor truy vấn. Thử cảnh/biến đổi khác."); return
    selected=st.selectbox("Query descriptor A",range(len(matches)),key=prefix+"query",
                          format_func=lambda i:f"#{i} · {matches[i].reason}")
    m=matches[selected]
    _plot(matching_view(a,b,pa,pb,matches,selected,presentation=_presentation()))
    if m.nearest is not None:
        st.write(f"Query A #{m.query} (x={pa[m.query].col_global:.1f},y={pa[m.query].row_global:.1f}) "
                 f"→ B #{m.nearest} (x={pb[m.nearest].col_global:.1f},y={pb[m.nearest].row_global:.1f}); "
                 f"nhì B #{m.second} (x={pb[m.second].col_global:.1f},y={pb[m.second].row_global:.1f}); "
                 f"d₁={m.d1:.5f}, d₂={m.d2:.5f}; "
                 f"ratio={m.ratio:.5f}" if m.ratio is not None else m.reason)
        error,correct=evaluate_geometric_correspondence(pa[m.query],pb[m.nearest],matrix)
        st.caption(f"Post-hoc affine error={error:.2f} px; {'phù hợp' if correct else 'không phù hợp'} "
                   "với transform đã biết (≤4 px). Ground truth KHÔNG chọn match.")
    st.write(f"Ratio test: {'ACCEPT' if m.accepted else 'REJECT'} · {m.reason}. "
             f"{sum(x.accepted for x in matches)}/{len(matches)} query qua ngưỡng {threshold:.2f}.")
    st.caption("Hai ảnh phát hiện/mô tả độc lập. Ratio pass không đảm bảo tương ứng đúng hình học.")


def _guided():
    step=int(st.session_state.sift_step)
    active="".join(f'<span class="{"done" if i<step else "active" if i==step else ""}"></span>' for i in range(1,19))
    st.markdown(f'<div class="chapter2-steps" aria-label="Bước {step}/18">{active}</div>',unsafe_allow_html=True)
    st.caption(f"BƯỚC {step}/18 · {STEPS[step-1]}")
    st.subheader(STEPS[step-1])
    name="Cảnh nhiễu" if step==6 else "Cấu trúc bất đối xứng" if step==7 else "Blob và góc rõ"
    image,pyramid,raw,filtered,described=_compute(name)
    preferred="low contrast" if step==6 else "edge" if step==7 else "kept"
    allowed=True
    if step==1:
        other,_=transform_scene(image,20,1.12)
        a,b=st.columns(2)
        with a: _plot(image_view(image,"Ảnh A",presentation=_presentation()))
        with b: _plot(image_view(other,"Cùng cảnh · quay 20° và scale 1.12",presentation=_presentation()))
        st.write("Điểm cục bộ phải tìm lại được khi cảnh đổi thang, góc và cường độ sáng — gần đúng, không tuyệt đối.")
    elif step==2:
        _plot(image_view(pyramid[0].dogs[1],"DoG = L(kσ) − L(σ) · signed",signed=True,presentation=_presentation()))
        _plot(image_view(image,"Cực trị strict 8+9+9 · chỉ là ứng viên",points=raw,presentation=_presentation()))
        st.write(f"{len(raw)} ứng viên lấy trực tiếp từ engine Scale Space; chưa tinh chỉnh/lọc/gán hướng.")
    elif step in (3,4,5,6,7,8):
        candidate,kp=_select_candidate(raw,filtered,f"sift_guided_candidate_{step}",preferred,
                                       show_status=step not in (6,7))
        if candidate is None: _nav(step,True); return
        if step in (3,4,5):
            _plot(image_view(image,"Ứng viên trên ảnh gốc",points=raw,selected=candidate,presentation=_presentation()))
            if step==3:
                _plot(response_patch(pyramid[candidate.octave].dogs,candidate,presentation=_presentation()))
                st.write(f"Lưới: (x={candidate.col},y={candidate.row},D={candidate.layer}); "
                         "vị trí liên tục được ước lượng bằng Taylor bậc hai, không bịa offset.")
            else:
                if step==4:
                    t=calculate_dog_derivatives(pyramid[candidate.octave].dogs,candidate.layer,candidate.row,candidate.col)
                    _plot(response_patch(pyramid[candidate.octave].dogs,candidate,presentation=_presentation()))
                    st.write(f"g=(Dx,Dy,Ds)={np.array2string(t.gradient,precision=6)}")
                    _matrix(t.hessian,("x","y","s"))
                    st.caption("Hessian 3×3 của DoG, khác Hessian không gian 2×2 ở bước lọc cạnh.")
                else: _taylor(pyramid,candidate,kp)
            if step==3: allowed=_gate(step,"Cực trị mẫu chỉ cho điểm bắt đầu; vị trí nội suy có thể lệch.",(candidate.octave,candidate.layer,candidate.row,candidate.col))
            if step==5: allowed=_gate(step,"Nếu độ lệch vượt nửa ô, chuyển sang ô lân cận và giải lại; có thể thất bại.",(candidate.octave,candidate.layer,candidate.row,candidate.col))
        elif step==6:
            _plot(image_view(image,"Ứng viên trước khi reveal",points=[candidate],selected=candidate,presentation=_presentation()))
            st.write(f"D thô={candidate.dog_value:+.6f}; ngưỡng |D̂|=.008000.")
            st.caption("Ngưỡng trong đơn vị DoG ảnh [0,1], không chia cho s=3; không đồng nhất với OpenCV contrastThreshold.")
            allowed=_gate(step,"Điểm yếu không ổn định trước nhiễu; |D̂| < T bị loại.",(candidate.octave,candidate.layer,candidate.row,candidate.col))
            if allowed:
                st.write(f"|D̂|={abs(kp.response):.6f} → {kp.status.upper()} · {kp.reason}")
                _plot(image_view(image,"Sau lọc tương phản và cạnh",points=filtered,selected=kp,presentation=_presentation()))
        elif step==7:
            h=calculate_spatial_hessian(pyramid[kp.octave].dogs,kp.layer,kp.row,kp.col)
            keep,tr,det,ratio=edge_decision(h)
            _plot(response_patch(pyramid[kp.octave].dogs,kp,presentation=_presentation()))
            _matrix(h,("x","y"))
            st.write(f"Dxx={h[0,0]:+.6f}, Dxy={h[0,1]:+.6f}, Dyy={h[1,1]:+.6f}; "
                     f"Tr={tr:+.6f}, Det={det:+.8f}; ngưỡng r=10 → 12.1.")
            allowed=_gate(step,"Dùng Hessian không gian 2×2 của DoG, không phải tensor Harris.",(candidate.octave,candidate.layer,candidate.row,candidate.col))
            if allowed:
                st.write(f"Tr²/Det={ratio:.4f}; {'KEEP' if keep else 'REJECT'} · {kp.reason}")
        else:
            _counts(filtered,raw)
            _plot(image_view(image,"Raw → refined → contrast → edge → kept",points=filtered,selected=kp,presentation=_presentation()))
            st.write(f"#{raw.index(candidate)}: {kp.status} · {kp.reason}")
    elif step in (9,10,11,12,13,14,15):
        if not described: st.warning("Không có descriptor hợp lệ."); _nav(step,True); return
        if step == 11:
            locations = {(p.octave,p.layer,p.row_local,p.col_local) for p in described}
            st.caption(f"Trong preset này: {len(locations)} vị trí keypoint có descriptor hợp lệ → "
                       f"{len(described)} bản ghi hướng/descriptor. Một vị trí có thể sinh nhiều hướng.")
        idx=st.selectbox("Keypoint đã giữ + hướng",range(len(described)),key=f"sift_guided_kp_{step}",
                         format_func=lambda i:f"#{i} · ({described[i].col_global:.1f},{described[i].row_global:.1f}) "
                         f"σ={described[i].sigma_global:.2f}, θ={described[i].orientation:.0f}°")
        oriented=described[idx]
        if step==9:
            other,_=transform_scene(image,25)
            a,b=st.columns(2)
            with a: _plot(image_view(image,"Cấu trúc gốc",points=[oriented],presentation=_presentation()))
            with b: _plot(image_view(other,"Ảnh quay +25° · nội suy làm thay đổi gradient",presentation=_presentation()))
            st.caption("Hướng của gradient và frame descriptor đổi cùng nhau gần đúng; phải phát hiện lại trên ảnh B.")
        elif step in (10,11):
            info=_orientation(pyramid,oriented,show_sample=step==10)
            if step==11: st.write(f"{len(info.peaks)} orientation record(s) ở cùng điểm/thang. "
                                  "Một cấu trúc đối xứng có thể có nhiều hướng gần bằng nhau.")
            if step==10: allowed=_gate(step,"Gradient tính từ Gaussian gần scale; atan2 giữ đúng bốn góc phần tư.",idx)
        elif step==12:
            patch,valid=sample_oriented_descriptor_patch(oriented,pyramid[oriented.octave])
            _plot(oriented_patch_view(patch,presentation=_presentation()))
            st.write(f"θ*={oriented.orientation:.0f}° · σ_local={oriented.sigma_local:.2f} · "
                     f"{valid.sum()}/{valid.size} mẫu hợp lệ.")
            st.caption("16×16 ở slide là sơ đồ canonical; support trên ảnh thật thay đổi theo σ.")
        elif step in (13,14):
            desc=_descriptor(pyramid,oriented,prefix=f"sift_guided_{step}_")
            if step==14:
                _plot(vector_view(desc.raw,"128D thô · 16 ô × 8 hướng",presentation=_presentation()))
                allowed=_gate(step,"Các 128 số là tổng gradient đã nội suy, không phải thanh trang trí.",idx)
        else:
            desc=build_sift_descriptor(oriented,pyramid[oriented.octave])
            stage=st.select_slider("Giai đoạn",options=("Thô","L2 lần 1","Clip 0.2","L2 lần 2"),key="sift_normal_stage")
            values=dict(zip(("Thô","L2 lần 1","Clip 0.2","L2 lần 2"),
                            (desc.raw,desc.first_l2,desc.clipped,desc.final)))[stage]
            _plot(vector_view(values,f"{stage} · ||d||₂={np.linalg.norm(values):.5f}",presentation=_presentation()))
            st.caption("Clip áp dụng sau chuẩn hóa lần đầu; sau L2 lần hai, một thành phần CÓ THỂ >0.2. "
                       "Không bảo đảm bất biến ánh sáng phổ quát.")
    elif step in (16,17):
        match_data=_matching("Cảnh biến đổi affine",20.,1.0,0.,1.,0.)
        if step==16:
            a,b,_,pa,pb=match_data
            left,right=st.columns(2)
            with left: _plot(image_view(a,f"Ảnh A · {len(pa)} descriptor",presentation=_presentation()))
            with right: _plot(image_view(b,f"Ảnh B · {len(pb)} descriptor",presentation=_presentation()))
            allowed=_gate(step,"Khoảng cách Euclid chỉ đo descriptor; kiểm tra hình học là bước hậu kiểm.",1)
            if allowed: _matching_panel(match_data,prefix="sift_guided_16_")
        else:
            threshold=st.slider("Lowe ratio threshold",.5,1.0,.8,.01,key="sift_guided_ratio")
            allowed=_gate(step,"Điều kiện strict d₁/d₂ < threshold; bằng ngưỡng bị loại.",threshold)
            if allowed: _matching_panel(match_data,threshold,prefix="sift_guided_17_")
    else:
        _counts(filtered,raw)
        _plot(image_view(image,"Ứng viên → keypoint sau lọc",points=filtered,presentation=_presentation()))
        st.write("DoG → Taylor/Hessian 3×3 → lọc contrast/Hessian 2×2 → hướng 36 bin → "
                 "descriptor 4×4×8 → L2/clip/L2 → hai láng giềng Euclid → Lowe ratio.")
        st.caption("Xấp xỉ rời rạc, biến đổi lớn, ảnh lặp, biên, nhiễu và chi phí tính toán đều là giới hạn; "
                   "ratio pass không bảo đảm match đúng. OpenCV SIFT không được dùng để tạo các số trong bài.")
        st.info("Tiếp theo: ORB-style dùng FAST + so sánh intensity nhị phân, Hamming thay cho "
                "histogram gradient 128D + Euclid. Cả hai cần kiểm tra match trên dữ liệu cụ thể.")
        allowed=_gate(step,"Tính bền vững là gần đúng; interpolation, quantization và occlusion có thể phá correspondence.",1)
    _nav(step,allowed)


def _lab_a():
    st.subheader("Lab A · DoG, Taylor và lọc")
    with st.expander("Ảnh và tham số pipeline",expanded=False):
        name=st.selectbox("Preset",list(PRESET_GOAL),key="sift_lab_a_preset")
        intervals=st.slider("Số khoảng mỗi octave",2,5,3,key="sift_lab_a_intervals")
        sigma=st.slider("σ base",1.0,2.5,1.6,.1,key="sift_lab_a_sigma")
        contrast=st.slider("Ngưỡng |D̂| · đơn vị [0,1]",0.,.04,.008,.001,key="sift_lab_a_contrast")
        edge=st.slider("r edge",2.,20.,10.,.5,key="sift_lab_a_edge")
    st.caption(PRESET_GOAL[name]+" Hỏi: ứng viên nào sống sau cả hai phép lọc?")
    signature=(name,intervals,sigma,contrast,edge)
    if st.session_state.get("sift_lab_a_signature") != signature:
        st.session_state.sift_lab_a_signature=signature
        st.session_state.pop("sift_lab_a_candidate",None)
        st.session_state.pop("sift_lab_a_reveal",None)
    image,pyramid,raw,filtered,_=_compute(name,intervals,sigma,contrast,edge,False)
    _counts(filtered,raw)
    candidate,kp=_select_candidate(raw,filtered,"sift_lab_a_candidate")
    _plot(image_view(image,"Trạng thái ứng viên tính thật",points=filtered,selected=kp,presentation=_presentation()))
    if candidate is not None:
        guess=st.radio("Dự đoán điểm này",("KEEP","REJECT"),index=None,key="sift_lab_a_guess",horizontal=True)
        if st.button("Reveal Taylor và quyết định",disabled=guess is None,key="sift_lab_a_reveal_button"):
            st.session_state.sift_lab_a_reveal=True
        if st.session_state.get("sift_lab_a_reveal"):
            _taylor(pyramid,candidate,kp)
            h=calculate_spatial_hessian(pyramid[kp.octave].dogs,kp.layer,kp.row,kp.col)
            keep,tr,det,ratio=edge_decision(h,edge)
            st.write(f"Hessian DoG 2×2: Dxx={h[0,0]:+.6f}, Dxy={h[0,1]:+.6f}, "
                     f"Dyy={h[1,1]:+.6f}, Tr={tr:+.6f}, Det={det:+.7f}, ratio={ratio:.3f}; "
                     f"edge check={'KEEP' if keep else 'REJECT'}.")
            st.info(f"Kết quả pipeline: {kp.status} · {kp.reason}")
    if st.button("Reset Lab A"):
        _reset("sift_lab_a_"); st.rerun()


def _lab_b():
    st.subheader("Lab B · Hướng và descriptor")
    name=st.selectbox("Preset Lab B",list(PRESET_GOAL),key="sift_lab_b_preset")
    image,pyramid,raw,filtered,described=_compute(name)
    if not described:
        st.warning("Không có descriptor ở cảnh này; chọn preset khác."); return
    signature=name
    if st.session_state.get("sift_lab_b_signature") != signature:
        st.session_state.sift_lab_b_signature=signature
        st.session_state.pop("sift_lab_b_point",None)
        st.session_state.pop("sift_lab_b_cell_row",None)
        st.session_state.pop("sift_lab_b_cell_col",None)
        st.session_state.pop("sift_lab_b_rotation_ready",None)
        st.session_state.pop("sift_lab_b_orientation_sample",None)
    idx=st.selectbox("Keypoint đã giữ + orientation",range(len(described)),key="sift_lab_b_point",
                     format_func=lambda i:f"#{i} · x={described[i].col_global:.1f}, y={described[i].row_global:.1f}, θ={described[i].orientation:.0f}°")
    kp=described[idx]
    _plot(image_view(image,"Keypoint đang phân tích",points=[kp],selected=kp,presentation=_presentation()))
    _orientation(pyramid,kp,show_sample=True,sample_key="sift_lab_b_orientation_sample")
    desc=_descriptor(pyramid,kp,prefix="sift_lab_b_")
    stage=st.select_slider("Chuẩn hóa",options=("Thô","L2 lần 1","Clip 0.2","L2 lần 2"),key="sift_lab_b_stage")
    values=dict(zip(("Thô","L2 lần 1","Clip 0.2","L2 lần 2"),
                    (desc.raw,desc.first_l2,desc.clipped,desc.final)))[stage]
    _plot(vector_view(values,stage,presentation=_presentation()))
    angle=st.slider("Quay ảnh để đối chiếu",-45,45,20,5,key="sift_lab_b_rotation")
    if st.button("Tính lại ảnh quay",key="sift_lab_b_rotate_compute"):
        st.session_state.sift_lab_b_rotation_ready=angle
    if "sift_lab_b_rotation_ready" in st.session_state:
        actual_angle=st.session_state.sift_lab_b_rotation_ready
        rotated,matrix,rotated_points=_rotation_compare(name,actual_angle)
        xy=matrix@np.array([kp.col_global,kp.row_global,1.])
        nearest=min(rotated_points,key=lambda p:np.hypot(p.col_global-xy[0],p.row_global-xy[1]),default=None)
        _plot(image_view(rotated,f"Quay {actual_angle}° · keypoint phát hiện lại",points=rotated_points,
                         selected=nearest,presentation=_presentation()))
        if nearest is not None:
            dist=float(np.hypot(nearest.col_global-xy[0],nearest.row_global-xy[1]))
            st.write(f"Vị trí affine dự kiến ({xy[0]:.2f},{xy[1]:.2f}); detection gần nhất "
                     f"cách {dist:.2f} px; θ gốc={kp.orientation:.0f}°, θ mới={nearest.orientation:.0f}°. "
                     "So sánh modulo 360, nội suy/biên có thể gây sai khác.")
    if st.button("Reset Lab B"):
        _reset("sift_lab_b_"); st.rerun()


def _lab_c():
    st.subheader("Lab C · Feature matching")
    with st.expander("Cảnh và biến đổi ảnh B",expanded=False):
        name=st.selectbox("Cảnh",("Cảnh biến đổi affine","Hoa văn lặp","Blob và góc rõ","Cảnh nhiễu"),key="sift_lab_c_name")
        angle=st.slider("Quay (độ)",-45,45,20,5,key="sift_lab_c_angle")
        scale=st.slider("Scale",.75,1.25,1.0,.05,key="sift_lab_c_scale")
        brightness=st.slider("Brightness",-.2,.2,0.,.02,key="sift_lab_c_brightness")
        gain=st.slider("Contrast multiplier",.6,1.4,1.,.05,key="sift_lab_c_gain")
        noise=st.slider("Noise σ",0.,.12,0.,.01,key="sift_lab_c_noise")
    ratio=st.slider("Lowe ratio threshold",.5,1.,.8,.01,key="sift_lab_c_ratio")
    signature=(name,angle,scale,brightness,gain,noise)
    if st.session_state.get("sift_lab_c_signature") != signature:
        st.session_state.sift_lab_c_signature=signature
        st.session_state.pop("sift_lab_c_computed",None)
        st.session_state.pop("sift_lab_c_query",None)
    st.caption(PRESET_GOAL[name]+" Noise dùng seed=2026; slider ratio chỉ tái lọc khoảng cách đã tính.")
    if st.button("Tính keypoints và descriptors A/B",key="sift_lab_c_compute"):
        st.session_state.sift_lab_c_computed=True
    if st.session_state.get("sift_lab_c_computed"):
        with st.spinner("Đang tính Gaussian/DoG và descriptor cho hai ảnh..."):
            data=_matching(name,angle,scale,brightness,gain,noise)
        st.write(f"Ảnh A: {len(data[3])} descriptors; ảnh B: {len(data[4])} descriptors.")
        _matching_panel(data,ratio,prefix="sift_lab_c_")
    else:
        st.info("Bấm nút Tính để tạo hai tập descriptor; thay đổi transform sẽ xóa kết quả cũ.")
    if st.button("Reset Lab C"):
        _reset("sift_lab_c_"); st.rerun()


def render():
    _init()
    st.session_state.sift_plot_count=0
    if not _presentation():
        st.markdown('<div class="chapter-kicker">CHAPTER 2 · INTERACTIVE FEATURE LAB</div>',unsafe_allow_html=True)
    st.title("SIFT · Learning Lab" if _presentation() else "SIFT · từ DoG đến descriptor 128D")
    if not _presentation():
        st.caption("Quan sát → dự đoán → tính → reveal → giải thích → thử nghiệm · bài tiếp nối Scale Space")
    left,right=st.columns([3,1])
    with left:
        st.radio("Không gian học",("Guided Tutorial","Experiment Lab"),key="sift_mode",horizontal=True,on_change=_save_view)
    with right:
        st.toggle("Presentation Mode",key="sift_presentation",on_change=_save_view)
    if st.session_state.sift_mode=="Guided Tutorial":
        _guided()
    else:
        tab_a,tab_b,tab_c=st.tabs(("A · Detection","B · Orientation/Descriptor","C · Matching"))
        with tab_a: _lab_a()
        with tab_b: _lab_b()
        with tab_c: _lab_c()
