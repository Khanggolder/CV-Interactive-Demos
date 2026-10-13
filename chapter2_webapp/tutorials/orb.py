"""Guided FAST/BRIEF/ORB-style learning with isolated orb_* state."""

from __future__ import annotations

from html import escape

import numpy as np
import streamlit as st

from utils.orb import (PRESET_GOALS, apply_fast_nms, bit_entropy,
    build_orb_image_pyramid,
    build_oriented_brief_descriptor, calculate_hamming_distance,
    calculate_intensity_moments, detect_fast_candidates, detect_fast_corner,
    generate_brief_test_pattern, map_pyramid_coordinates, match_binary_descriptors,
    microscope_fixtures, orb_pipeline, pattern_learning_demo, quantize_orientation,
    rank_fast_candidates_with_harris, teaching_presets, transform_scene,
    unpack_binary_descriptor)
from utils.orb_visualization import (bits_view,brief_pair_view,centroid_view,
    fast_circle,image_view,matching_view,score_view)

STEPS=(
    "Vì sao cần ORB?", "Kiến trúc oFAST + rBRIEF", "Vòng FAST 16 pixel",
    "Dự đoán góc", "Pretest bốn pixel", "Cung FAST-12 đầy đủ",
    "FAST score → NMS → Harris", "Kim tự tháp ảnh ORB",
    "Vì sao keypoint cần hướng?", "Moment và intensity centroid",
    "Thí nghiệm quay ảnh", "Một bit BRIEF", "Tích lũy 256 bit",
    "BRIEF và rBRIEF", "Chọn test pattern", "XOR và Hamming",
    "Ghép descriptor", "ORB-style so với SIFT",
)
OBJECTIVES=(
    "So sánh payload descriptor, không suy ra một tốc độ cố định.",
    "Nối detection → orientation → description → matching.",
    "Đọc đúng thứ tự 16 pixel và dấu ngưỡng strict.",
    "Dự đoán từ cung liên tiếp, không chỉ đếm tổng.",
    "Phân biệt điều kiện cần với điều kiện đủ.",
    "Thấy cung có thể quấn từ vị trí 16 về 1.",
    "Tách eligibility FAST, score, NMS và Harris ranking.",
    "Theo dõi scale và ánh xạ tâm pixel về ảnh gốc.",
    "FAST tự thân không cho hướng chính.",
    "Tính m00, m10, m01 và atan2 trên patch tròn.",
    "Đối chiếu đúng keypoint tương ứng dưới phép quay đã biết.",
    "Một bit là phép so strict trên pixel đã Gaussian blur.",
    "256 phép thử thật tạo 256 bit = 32 byte.",
    "Xoay cặp test trong hệ (x phải,y xuống).",
    "Entropy/correlation giúp chọn cặp; demo học không phải ORB chuẩn.",
    "Hamming chính xác là popcount của XOR.",
    "Match theo descriptor và đúng hình học là hai nhãn khác nhau.",
    "So sánh phương pháp, dung lượng và giới hạn trung thực.",
)
GATES={
    4:("Cung đủ 12 pixel cùng cực tính?",("CORNER","NOT CORNER")),
    5:("Qua pretest bốn pixel có chắc là góc?",("Có","Không")),
    6:("Cung có được phép đi từ 16 sang 1?",("Có","Không")),
    10:("Nếu m10<0 và m01>0, góc thuộc góc phần tư II theo y xuống?",("Có","Không")),
    12:("I(p)<I(q) cho bit bằng 1?",("Có","Không")),
    16:("Hamming đếm bit khác nhau, không dùng Euclid?",("Có","Không")),
    17:("Match được chấp nhận có chắc đúng hình học?",("Có","Không")),
    18:("ORB-style trong lab dùng đúng learned pattern OpenCV?",("Có","Không")),
}


@st.cache_data(show_spinner=False,max_entries=20)
def _pipeline(name,levels=8,threshold=20,describe=True):
    return orb_pipeline(teaching_presets()[name],threshold=threshold,levels=levels,describe=describe)


@st.cache_data(show_spinner=False,max_entries=12)
def _pair(name,angle,scale,brightness,contrast,noise):
    a=teaching_presets()[name]
    b,matrix=transform_scene(a,angle,scale,brightness,contrast,noise)
    pa,ka,da=orb_pipeline(a)
    pb,kb,db=orb_pipeline(b)
    return a,b,matrix,pa,ka,da,pb,kb,db


@st.cache_data(show_spinner=False,max_entries=2)
def _learning():
    return pattern_learning_demo()


@st.cache_data(show_spinner=False,max_entries=32)
def _lab_a_level(name,level,threshold,arc_length):
    pyramid=build_orb_image_pyramid(teaching_presets()[name])
    selected=pyramid[level]
    eligible,scores=detect_fast_candidates(selected.image,threshold,arc_length)
    nms=apply_fast_nms(scores,eligible)
    ranked,harris=rank_fast_candidates_with_harris(selected.image,nms,scores)
    return selected,eligible,scores,nms,ranked,harris


def _init():
    s=st.session_state
    s.setdefault("orb_step",1)
    s.setdefault("orb_mode",s.get("orb_saved_mode","Guided Tutorial"))
    s.setdefault("orb_presentation",s.get("orb_saved_presentation",False))


def _save_view():
    st.session_state.orb_saved_mode=st.session_state.orb_mode
    st.session_state.orb_saved_presentation=st.session_state.orb_presentation


def _reset(prefix="orb_"):
    keep={"orb_mode","orb_presentation","orb_saved_mode","orb_saved_presentation"}
    for key in list(st.session_state):
        if key.startswith(prefix) and key not in keep:del st.session_state[key]
    st.session_state.orb_step=1


def _reset_lab(prefix):
    for key in list(st.session_state):
        if key.startswith(prefix):del st.session_state[key]


def _move(delta):
    st.session_state.orb_step=max(1,min(18,st.session_state.orb_step+delta))


def _plot(fig):
    index=st.session_state.get("orb_plot_count",0)
    st.session_state.orb_plot_count=index+1
    st.plotly_chart(fig,key=f"orb_plot_{index}",use_container_width=True,
                    config={"displayModeBar":False})


def _present():return bool(st.session_state.orb_presentation)


def _gate(step,answer,explanation,signature=None):
    if signature is not None and st.session_state.get(f"orb_sig_{step}")!=signature:
        st.session_state[f"orb_sig_{step}"]=signature
        st.session_state.pop(f"orb_guess_{step}",None)
        st.session_state.pop(f"orb_reveal_{step}",None)
    question,options=GATES[step]
    st.markdown(f'<div class="question">DỰ ĐOÁN · {escape(question)}</div>',unsafe_allow_html=True)
    guess=st.radio("Chọn trước khi reveal",options,index=None,key=f"orb_guess_{step}",horizontal=True)
    if st.button("Reveal kết quả",key=f"orb_button_{step}",disabled=guess is None):
        st.session_state[f"orb_reveal_{step}"]=True
    revealed=bool(st.session_state.get(f"orb_reveal_{step}",False))
    if revealed:st.success(("Đúng. " if guess==answer else "Chưa đúng. ")+explanation)
    return revealed


def _nav(step,allowed):
    a,b,c=st.columns(3)
    a.button("← Previous",on_click=_move,args=(-1,),disabled=step==1,width="stretch")
    b.button("Next →",on_click=_move,args=(1,),disabled=step==18 or not allowed,width="stretch")
    c.button("Reset",on_click=_reset,width="stretch")


def _default_point(points,key):
    if not points:return None
    index=st.selectbox("Keypoint đã giữ",range(len(points)),index=1 if len(points)>1 else 0,key=key,
        format_func=lambda i:f"#{i} · tầng {points[i].level} · (x={points[i].col}, y={points[i].row}) · FAST {points[i].fast_score:.1f}")
    return points[index]


def _scene_data():
    name="Cấu trúc bất đối xứng"
    image=teaching_presets()[name]
    levels,points,diag=_pipeline(name)
    return image,levels,points,diag


def _keypoint_descriptor(levels,point,*,steering=True,quantized=True):
    level=levels[point.level]
    moments=calculate_intensity_moments(level.image,point.row,point.col)
    brief=build_oriented_brief_descriptor(level.image,point.row,point.col,moments.angle,
        steering=steering,quantized=quantized)
    return level,moments,brief


def _fast_inspector(image,row,col,threshold,arc_length=12,*,replay=16,reveal=False):
    result=detect_fast_corner(image,row,col,threshold,arc_length)
    if not result.valid:
        st.warning("Tâm quá gần biên: vòng bán kính 3 không nằm trọn trong ảnh.")
        return None
    _plot(fast_circle(image,row,col,result,shown=replay,show_arc=reveal,presentation=_present()))
    st.caption(f"Iₚ={result.center:.1f}; t={threshold:.1f} (0–255). Sáng nếu I>Iₚ+t, tối nếu I<Iₚ−t. "
               "Viền đậm là bốn điểm pretest. Màu xanh=sáng, đỏ=tối, xám=tương tự.")
    i=st.slider("Soi pixel vòng #",1,16,1,key=f"orb_ring_pixel_{st.session_state.orb_step}_{arc_length}")
    st.write(f"#{i}: I={result.values[i-1]:.1f}; Δ=I−Iₚ={result.values[i-1]-result.center:+.1f}; "
             f"{result.labels[i-1]}.")
    return result


def _moments_panel(level,point,moments,prefix):
    radius=moments.patch.shape[0]//2
    coords=[(int(x),int(y)) for y in range(-radius,radius+1) for x in range(-radius,radius+1)
            if moments.mask[y+radius,x+radius]]
    selected=st.selectbox("Pixel moment",coords,key=prefix+"moment_pixel",
                          format_func=lambda v:f"(Δx={v[0]}, Δy={v[1]})")
    _plot(centroid_view(moments,selected=selected,presentation=_present()))
    x,y=selected;intensity=float(moments.patch[y+radius,x+radius])
    st.write(f"I={intensity:.2f}; x·I={x*intensity:+.2f}; y·I={y*intensity:+.2f}.")
    st.write(f"m00={moments.m00:.2f}; m10={moments.m10:+.2f}; m01={moments.m01:+.2f}.")
    if moments.angle is None:st.warning("Centroid/hướng không xác định: tổng hoặc vector moment quá nhỏ.")
    else:st.write(f"C=({moments.cx:+.3f},{moments.cy:+.3f}); θ=atan2(m01,m10)={moments.angle:.2f}°.")
    st.caption("Patch tròn bán kính 9 pixel; x sang phải, y xuống. Độ sáng nền và cắt biên có thể đổi centroid.")


def _brief_panel(level,point,brief,prefix,*,show_full=False):
    if brief.packed is None:
        st.warning("Descriptor không hợp lệ: "+brief.reason);return
    index=st.slider("Cặp BRIEF #",0,255,0,key=prefix+"pair")
    sample=brief.samples[index]
    _plot(brief_pair_view(level.image,point.row,point.col,sample,presentation=_present()))
    st.write(f"p=({sample[0]:.0f},{sample[1]:.0f}), I(p)={sample[4]:.3f}; "
             f"q=({sample[2]:.0f},{sample[3]:.0f}), I(q)={sample[5]:.3f} → "
             f"bit {index} = {int(brief.bits[index])}.")
    if show_full:
        _plot(bits_view(brief.bits,selected=index,presentation=_present()))
        st.code(" ".join(f"{byte:02x}" for byte in brief.packed))
        byte=index//8
        st.caption(f"Byte {byte} = {brief.packed[byte]:08b}; MSB là bit {byte*8}. "
                   "256 so sánh thật → 32 byte. Pattern deterministic seed=2026, không phải learned pattern OpenCV.")


def _guided():
    step=st.session_state.orb_step
    st.markdown(f'<div class="chapter-kicker">GUIDED · BƯỚC {step}/18</div>',unsafe_allow_html=True)
    st.subheader(f"{step:02d} · {STEPS[step-1]}")
    st.caption("Mục tiêu: "+OBJECTIVES[step-1])
    st.markdown('<div class="chapter2-steps">'+''.join(
        f'<span class="{"active" if i==step else "done" if i<step else ""}"></span>'
        for i in range(1,19))+'</div>',unsafe_allow_html=True)
    allowed=True
    fixtures=microscope_fixtures()
    if step==1:
        method=st.selectbox("Quan sát descriptor",("SIFT float32","ORB-style binary"),key="orb_g_intro")
        st.metric("Payload / keypoint", "512 byte" if method.startswith("SIFT") else "32 byte")
        st.progress(1. if method.startswith("SIFT") else 1/16)
        st.write("128 × float32 = 512 byte; 256 bit = 32 byte. Tỉ lệ payload 16:1, không tính metadata.")
        st.caption("Thời gian chạy không có một tỉ lệ cố định; phụ thuộc ảnh, máy và triển khai.")
    elif step==2:
        stage=st.select_slider("Theo dõi chuỗi phép tính",options=("FAST","NMS/Harris","Pyramid","Centroid","BRIEF","Hamming"),key="orb_g_stage")
        st.info({"FAST":"16 pixel thật → cung 12 cùng cực tính.","NMS/Harris":"FAST score tìm cực đại; Harris chỉ xếp hạng điểm đã qua FAST.",
                 "Pyramid":"Nhiều kích thước ảnh, không phải DoG của SIFT.","Centroid":"m10,m01 → atan2; FAST chưa cho hướng.",
                 "BRIEF":"So sánh cặp cường độ đã blur; xoay cặp theo hướng.","Hamming":"XOR 32 byte rồi đếm bit 1."}[stage])
        _plot(image_view(fixtures["Cung 12"],"Đầu vào thật · fixture FAST",selected=(7,7),presentation=_present()))
    elif step in (3,4,5,6):
        names={3:("Cung 12","Vùng phẳng","Cạnh thẳng"),4:("Cung 12","Vùng phẳng","Cạnh thẳng"),
               5:("Pretest sống, cung rớt","Early reject"),6:("Wrap 16→1","Cung 12")}[step]
        name=st.selectbox("Patch kiểm chứng",names,key=f"orb_g_fixture_{step}")
        threshold=st.slider("Ngưỡng t (intensity)",0,100,20,key=f"orb_g_threshold_{step}")
        replay=st.slider("Đã kiểm tra đến pixel #",0,16,16 if step!=6 else 4,key="orb_g_replay_6") if step==6 else 16
        result=_fast_inspector(fixtures[name],7,7,threshold,replay=replay,
                               reveal=bool(st.session_state.get(f"orb_reveal_{step}")))
        if step==3:
            st.info("Thứ tự 1 trên, 5 phải, 9 dưới, 13 trái; dãy vòng là cyclic. Chọn pixel để kiểm tra giá trị.")
        elif step==4:
            answer="CORNER" if result.accepted else "NOT CORNER"
            allowed=_gate(step,answer,f"Cung tốt nhất có margin {result.score:.1f}; ngưỡng {threshold}; "
                          f"FAST-12 {'nhận' if result.accepted else 'loại'}.",(name,threshold))
        elif step==5:
            allowed=_gate(step,"Không",f"Pretest {'qua' if result.early_pass else 'loại sớm'}, "
                          f"nhưng full FAST-12 {'nhận' if result.accepted else 'loại'}; "
                          "qua pretest không đủ để kết luận.",(name,threshold))
        else:
            allowed=_gate(step,"Có",f"Cung tốt nhất: {[i+1 for i in result.arc]}; "
                          f"chuỗi dài nhất tại t={threshold}: {result.longest}.",(name,threshold))
            if allowed:st.write("Các margin trên cung:",list(np.round(result.margins,2)))
    elif step==7:
        image,levels,points,diag=_scene_data()
        level=st.slider("Tầng pyramid",0,len(levels)-1,2,key="orb_g_score_level")
        info=diag[level]
        ids=np.argwhere(info["eligible"])
        if len(ids):
            index=st.slider("Ứng viên FAST",0,len(ids)-1,0,key="orb_g_score_candidate")
            row,col=map(int,ids[index]);selected=(row,col)
        else:selected=None
        _plot(score_view(info["scores"],info["eligible"],info["nms"],selected,presentation=_present()))
        st.write(f"FAST eligible={int(info['eligible'].sum())}; NMS kept={int(info['nms'].sum())}; "
                 f"Harris-ranked={len(info['ranked'])}.")
        if selected:
            r,c=selected
            retained=any(rr==r and cc==c for rr,cc,_,_ in info["ranked"])
            st.write(f"(x={c},y={r}): FAST score={info['scores'][r,c]:.3f}; "
                     f"NMS={'KEEP' if info['nms'][r,c] else 'REJECT'}; "
                     f"Harris={info['harris'][r,c]:+.7f}; final={'KEEP' if retained else 'REJECT'}.")
            st.caption("Lân cận NMS 3×3 (score theo hàng/cột): "+
                       np.array2string(info["scores"][r-1:r+2,c-1:c+2],precision=1))
        st.caption("Score = max các min margin trên mọi cung 12. NMS 3×3, hòa điểm chọn (row,col) nhỏ nhất; "
                   "Harris chỉ xếp hạng những điểm còn lại. Đây là thứ tự sư phạm, không khẳng định giống mọi ORB production.")
    elif step==8:
        image,levels,points,diag=_scene_data()
        index=st.slider("Tầng ảnh",0,len(levels)-1,1,key="orb_g_level")
        level=levels[index]
        local=[p for p in points if p.level==index]
        _plot(image_view(level.image,f"Tầng {index} · {level.image.shape[1]}×{level.image.shape[0]}",points=local,presentation=_present()))
        st.write(f"Scale danh nghĩa = 1.2^{index} = {level.nominal_scale:.3f}; "
                 f"tỉ lệ kích thước thực x={level.scale_x:.4f}, y={level.scale_y:.4f}; "
                 f"{int(diag[index]['eligible'].sum())} FAST candidates.")
        if local:
            p=local[0];st.caption(f"Ví dụ (x={p.col},y={p.row}) tầng → ({p.x_original:.2f},{p.y_original:.2f}) "
                                "ảnh gốc theo ánh xạ tâm pixel (x+0.5)·W0/Wl−0.5.")
    elif step in (9,10,11,12,13,14):
        image,levels,points,diag=_scene_data()
        if not points:st.warning("Không có keypoint; kiểm tra preset.");_nav(step,True);return
        point=_default_point(points,f"orb_g_point_{step}")
        level,moments,brief=_keypoint_descriptor(levels,point)
        if step==9:
            _plot(image_view(level.image,"FAST keypoint · chưa có hướng từ detector",selected=(point.row,point.col),presentation=_present()))
            st.info("FAST chỉ trả vị trí/score. Hướng θ bên dưới được tính riêng từ moment cường độ.")
            st.write(f"Tầng {point.level}; FAST score={point.fast_score:.2f}; θ centroid={moments.angle:.2f}°.")
        elif step==10:
            _moments_panel(level,point,moments,"orb_g_")
            allowed=_gate(step,"Có","atan2 giữ đúng góc phần tư; y ảnh hướng xuống.",point.level*10000+point.row*100+point.col)
        elif step==11:
            angle=st.slider("Quay ảnh theo chiều kim đồng hồ (°)",-60,60,20,key="orb_g_rotation")
            rotated,matrix=transform_scene(image,angle)
            rotated_levels,rotated_points,_=orb_pipeline(rotated)
            predicted=matrix@np.array([point.x_original,point.y_original,1.])
            if rotated_points:
                q=min(rotated_points,key=lambda p:(p.x_original-predicted[0])**2+(p.y_original-predicted[1])**2)
                displacement=float(np.hypot(q.x_original-predicted[0],q.y_original-predicted[1]))
                _plot(image_view(rotated,"Ảnh sau quay · keypoint gần tọa độ affine dự kiến",
                                 selected=(q.row,q.col),presentation=_present()))
                st.write(f"Tọa độ dự kiến=({predicted[0]:.2f},{predicted[1]:.2f}); "
                         f"điểm gần nhất cách {displacement:.2f}px; góc quay ảnh={angle:+.1f}°.")
                if displacement<=5:
                    delta=(q.angle-point.angle+180)%360-180
                    st.write(f"Ứng viên gần trong 5px: Δθ đo={delta:+.2f}° (chưa chứng minh cùng identity).")
                else:
                    st.warning("Không thấy keypoint tương ứng trong 5px; không so sánh hai hướng như cùng một điểm.")
                st.caption("Vị trí gần nhau chưa chứng minh cùng identity, đặc biệt tại cấu trúc lặp. "
                           "Interpolation, dịch keypoint và patch support làm Δθ lệch góc quay.")
            else:st.warning("Không có điểm FAST sau quay ở cấu hình này.")
        elif step==12:
            _brief_panel(level,point,brief,"orb_g_bit_")
            allowed=_gate(step,"Có","So sánh strict: bằng nhau cho 0; ảnh đã blur Gaussian σ=1.2.",
                          (point.level,point.row,point.col))
        elif step==13:
            count=st.slider("Số phép thử đã thực hiện",1,256,16,key="orb_g_bit_count")
            _plot(bits_view(brief.bits,selected=max(0,count-1),presentation=_present()))
            st.write("Bit đã tính:"," ".join("".join(map(str,brief.bits[i:i+8]))
                                          for i in range(0,count,8)))
            st.caption(f"{count}/256 phép thử hiển thị; descriptor đầy đủ có {len(brief.bits)} bit / "
                       f"{len(brief.packed)} byte. Byte đầu={brief.packed[0]:08b} (MSB trước).")
        else:
            steer=st.toggle("Xoay cặp BRIEF theo θ",value=True,key="orb_g_steer")
            quant=st.toggle("Lượng tử 30 hướng (12°)",value=True,key="orb_g_quant")
            trial=build_oriented_brief_descriptor(level.image,point.row,point.col,moments.angle,
                                                    steering=steer,quantized=quant)
            _brief_panel(level,point,trial,"orb_g_steer_",show_full=True)
            st.caption(f"θ đo={moments.angle:.2f}°; θ test={trial.steering_angle:.2f}°. "
                       "R(θ) trên (x phải,y xuống); lấy mẫu nearest bằng np.rint, sample ngoài ảnh → invalid.")
    elif step==15:
        pattern,bits,entropy,selected=_learning()
        choice=st.selectbox("Cặp pattern ứng viên",range(80),key="orb_g_pattern")
        q=float(bits[:,choice].mean())
        corr=0. if not len(selected) or np.std(bits[:,choice])==0 else float(np.corrcoef(bits[:,choice],bits[:,selected[0]])[0,1])
        st.bar_chart({"bit frequency q":bits.mean(axis=0)[:24],"entropy H(q)":entropy[:24]})
        st.write(f"Cặp #{choice}: {pattern[choice].astype(int).tolist()}; q={q:.3f}; "
                 f"H(q)={bit_entropy(q):.3f} bit; corr với cặp chọn đầu={corr:+.3f}.")
        st.write("12 cặp demo được chọn:",selected.tolist())
        st.caption("Educational learned-pattern demonstration: 40 patch synthetic, greedy ưu tiên entropy cao "
                   "và |correlation|<0.65. Descriptor runtime dùng pattern deterministic khác; KHÔNG phải ORB learned pattern.")
    elif step in (16,17):
        a,b,matrix,pa,ka,da,pb,kb,db=_pair("Cấu trúc bất đối xứng",20,1.,0,1.,0)
        if not ka or not kb:st.warning("Thiếu keypoint để ghép.");_nav(step,True);return
        method=st.selectbox("Quy tắc ghép",("cross-check","nearest","ratio"),key=f"orb_g_match_method_{step}")
        matches=match_binary_descriptors(ka,kb,method=method,transform=matrix)
        index=st.selectbox("Query descriptor A",range(len(matches)),key=f"orb_g_match_index_{step}")
        match=matches[index]
        if step==16:
            if match.train is not None:
                start=st.slider("Byte XOR cần soi",0,31,0,key="orb_g_xor_byte")
                x=ka[index].descriptor[start];y=kb[match.train].descriptor[start]
                st.code(f"A   {x:08b}\nB   {y:08b}\nXOR {x^y:08b}")
                st.caption(f"Byte này có {(x^y).bit_count()} bit khác; full descriptor dùng XOR + popcount trên 32 byte.")
                _plot(bits_view(unpack_binary_descriptor(ka[index].descriptor)^
                                unpack_binary_descriptor(kb[match.train].descriptor),selected=start*8,presentation=_present()))
                allowed=_gate(step,"Có",f"Hamming full 256 bit = {match.distance}; "
                              f"kiểm tra độc lập = {calculate_hamming_distance(ka[index].descriptor,kb[match.train].descriptor)}.",
                              (index,match.train,method))
        else:
            allowed=_gate(step,"Không",f"Accepted={match.accepted}; lỗi affine={match.geometric_error:.2f}px; "
                          f"đúng hình học (≤4px)={match.correct_geometry}. Ground truth chỉ dùng hậu kiểm.",
                          (index,method))
            revealed=allowed
            _plot(matching_view(a,b,ka,kb,matches,selected=index,
                                presentation=_present(),reveal=revealed))
            st.write(f"A#{index} → B#{match.train}: Hamming={match.distance}; "
                     f"second={match.second_distance}." +
                     (f" Rule: {match.reason}." if revealed else " Dự đoán trước khi xem rule."))
            if allowed:st.write(f"Tổng {sum(m.accepted for m in matches)} match accepted; "
                                f"{sum(m.accepted and m.correct_geometry is True for m in matches)} đúng hình học.")
    else:
        choice=st.radio("Chọn thuộc tính muốn ôn",("FAST","Centroid","BRIEF","Hamming","Bộ nhớ"),key="orb_g_recap",horizontal=True)
        text={"FAST":"16 pixel, cung 12 liên tiếp cùng cực tính; pretest chỉ loại sớm.",
              "Centroid":"m00,m10,m01; θ=atan2(m01,m10).",
              "BRIEF":"256 phép so strict trên ảnh blur → 32 byte; cặp xoay theo hướng.",
              "Hamming":"Đếm bit 1 của XOR; cross-check khác ratio test.",
              "Bộ nhớ":"SIFT 128 float32=512 byte; ORB-style 256 bit=32 byte, payload 16:1."}[choice]
        st.info(text)
        st.caption("Tiếp theo, Feature Matching Challenge đo acceptance, precision, coverage và "
                   "lỗi hình học trên cùng cặp ảnh; số keypoint/descriptor riêng lẻ chưa đủ kết luận.")
        allowed=_gate(step,"Không","Runtime pattern deterministic, không sao chép learned pattern OpenCV; "
                      "không bảo đảm scale/rotation invariance hay tốc độ cố định.",choice)
    _nav(step,allowed)


def _lab_a():
    st.subheader("Lab A · FAST Corner Explorer")
    with st.expander("Input / tham số FAST",expanded=not _present()):
        source=st.selectbox("Nguồn ảnh",("Microscope 15×15","Preset 129×129"),key="orb_a_source")
        if source.startswith("Microscope"):
            name=st.selectbox("Fixture",tuple(microscope_fixtures()),key="orb_a_fixture")
            image=microscope_fixtures()[name]
        else:
            name=st.selectbox("Preset",tuple(PRESET_GOALS),key="orb_a_preset")
            image=teaching_presets()[name];st.caption(PRESET_GOALS[name])
        row=st.slider("Tâm y=hàng",3,image.shape[0]-4,image.shape[0]//2,key="orb_a_row")
        col=st.slider("Tâm x=cột",3,image.shape[1]-4,image.shape[1]//2,key="orb_a_col")
        t=st.slider("Ngưỡng FAST t",0,100,20,key="orb_a_threshold")
        n=st.slider("Chiều dài cung n",3,16,12,key="orb_a_arc")
    result=detect_fast_corner(image,row,col,t,n)
    signature=(source,name,row,col,t,n)
    if st.session_state.get("orb_a_signature")!=signature:
        st.session_state.orb_a_signature=signature
        st.session_state.pop("orb_a_reveal",None)
        st.session_state.pop("orb_a_guess",None)
    guess=st.radio("Dự đoán",("CORNER","NOT CORNER"),index=None,key="orb_a_guess",horizontal=True)
    if st.button("Run pretest + full test / Reveal",disabled=guess is None,key="orb_a_run"):
        st.session_state.orb_a_reveal=True
    _plot(fast_circle(image,row,col,result,show_arc=bool(st.session_state.get("orb_a_reveal")),presentation=_present()))
    ring=st.slider("Soi vòng pixel #",1,16,1,key="orb_a_ring")
    st.write(f"#{ring}: I={result.values[ring-1]:.1f}; {result.labels[ring-1]}; "
             f"margin sáng={result.values[ring-1]-result.center:+.1f}.")
    if st.session_state.get("orb_a_reveal"):
        st.write(f"Pretest: {'PASS' if result.early_pass else 'EARLY REJECT' if n==12 else 'DISABLED'}; "
                 f"full arc={result.longest}; FAST score={result.score:.2f}; "
                 f"{'CORNER' if result.accepted else 'NOT CORNER'}.")
        st.caption(f"Cung tốt nhất: {[i+1 for i in result.arc]}; margins: {list(np.round(result.margins,1))}. "
                   "n≠12: pretest ≥3/4 không có bảo đảm, nên tắt; full test vẫn chạy.")
    if source.startswith("Preset"):
        levels=st.slider("Pyramid level",0,7,0,key="orb_a_level")
        selected,eligible,scores,nms,ranked,harris=_lab_a_level(name,levels,t,n)
        use_nms=st.toggle("Hiện điểm qua NMS",value=True,key="orb_a_nms")
        _plot(score_view(scores,eligible,nms if use_nms else eligible,
                         presentation=_present()))
        st.write(f"Tầng {levels}: {selected.image.shape[1]}×{selected.image.shape[0]}, "
                 f"FAST={int(eligible.sum())}, NMS={int(nms.sum())}, "
                 f"Harris-ranked={len(ranked)}.")
        if row<selected.image.shape[0] and col<selected.image.shape[1]:
            st.caption(f"Tọa độ level ({col},{row}) → original "
                       f"{map_pyramid_coordinates(selected,row,col)}.")
    st.button("Reset Lab A",on_click=_reset_lab,args=("orb_a_",))


def _lab_b():
    st.subheader("Lab B · Orientation & Binary Descriptor")
    name=st.selectbox("Ảnh",("Cấu trúc bất đối xứng","Bàn cờ","Cảnh có nhiễu"),key="orb_b_preset")
    levels,points,diag=_pipeline(name)
    if not points:st.warning("Không có keypoint ở preset này.");return
    point=_default_point(points,"orb_b_point")
    level,moments,brief=_keypoint_descriptor(levels,point)
    signature=(name,point.level,point.row,point.col)
    if st.session_state.get("orb_b_signature")!=signature:
        st.session_state.orb_b_signature=signature
        st.session_state.pop("orb_b_computed",None)
    _moments_panel(level,point,moments,"orb_b_")
    angle=st.slider("Quay ảnh (°)",-60,60,20,key="orb_b_angle")
    if st.button("Tính lại ảnh quay",key="orb_b_rotate"):
        st.session_state.orb_b_rotation_ready=(name,angle,point.level,point.row,point.col)
    if st.session_state.get("orb_b_rotation_ready")== (name,angle,point.level,point.row,point.col):
        rotated,matrix=transform_scene(teaching_presets()[name],angle)
        _,rotated_points,_=orb_pipeline(rotated)
        target=matrix@np.array([point.x_original,point.y_original,1.])
        if rotated_points:
            q=min(rotated_points,key=lambda p:(p.x_original-target[0])**2+(p.y_original-target[1])**2)
            error=float(np.hypot(q.x_original-target[0],q.y_original-target[1]))
            st.write(f"Vị trí affine dự kiến={target[:2].round(2)}; nearest detected="
                     f"({q.x_original:.2f},{q.y_original:.2f}), cách {error:.2f}px.")
            if error<=5:
                st.write(f"Ứng viên gần trong 5px (chưa chứng minh cùng identity): "
                         f"θ cũ={point.angle:.1f}°, θ mới={q.angle:.1f}°.")
            else:
                st.warning("Không tìm được keypoint tương ứng trong 5px; không so θ cũ/mới của hai điểm khác nhau.")
    steering=st.toggle("rBRIEF steering",value=True,key="orb_b_steer")
    quantized=st.toggle("30 hướng, mỗi 12°",value=True,key="orb_b_quant")
    current=build_oriented_brief_descriptor(level.image,point.row,point.col,moments.angle,
                                              steering=steering,quantized=quantized)
    if st.session_state.get("orb_b_computed") not in (None,(signature,steering,quantized)):
        st.session_state.pop("orb_b_computed",None)
    if st.button("Compute 256-bit descriptor",key="orb_b_compute"):
        st.session_state.orb_b_computed=(signature,steering,quantized)
    if st.session_state.get("orb_b_computed")== (signature,steering,quantized):
        _brief_panel(level,point,current,"orb_b_",show_full=True)
        st.write(f"θ steering={current.steering_angle:.1f}°; valid={current.packed is not None}.")
    else:st.info("Thay input/hướng làm mất kết quả cũ. Bấm Compute để xem bit và byte mới.")
    st.button("Reset Lab B",on_click=_reset_lab,args=("orb_b_",))


def _lab_c():
    st.subheader("Lab C · Matching & SIFT Comparison")
    with st.expander("Cảnh và phép biến đổi",expanded=not _present()):
        name=st.selectbox("Cảnh",("Cấu trúc bất đối xứng","Hoa văn lặp","Bàn cờ"),key="orb_c_scene")
        angle=st.slider("Quay (°)",-45,45,20,key="orb_c_angle")
        scale=st.slider("Scale",.7,1.3,1.,.05,key="orb_c_scale")
        brightness=st.slider("Brightness",-40,40,0,key="orb_c_brightness")
        contrast=st.slider("Contrast",.7,1.3,1.,.05,key="orb_c_contrast")
        noise=st.slider("Gaussian noise σ",0,30,0,key="orb_c_noise")
    signature=(name,angle,scale,brightness,contrast,noise)
    if st.session_state.get("orb_c_signature")!=signature:
        st.session_state.orb_c_signature=signature
        st.session_state.pop("orb_c_calculated",None)
    if st.button("Calculate A/B descriptors",key="orb_c_calculate"):
        st.session_state.orb_c_calculated=signature
    if st.session_state.get("orb_c_calculated")==signature:
        a,b,matrix,pa,ka,da,pb,kb,db=_pair(*signature)
        method=st.selectbox("Matching rule",("nearest","cross-check","ratio"),key="orb_c_rule")
        matches=match_binary_descriptors(ka,kb,method=method,transform=matrix)
        st.write(f"A: {len(ka)} descriptors; B: {len(kb)} descriptors; "
                 f"accepted: {sum(m.accepted for m in matches)}; "
                 f"accepted & đúng hình học ≤4px: {sum(m.accepted and m.correct_geometry is True for m in matches)}.")
        if matches:
            index=st.selectbox("Soi query A",range(len(matches)),key="orb_c_query")
            m=matches[index]
            st.write(f"Query {index}: B#{m.train}, Hamming={m.distance}, second={m.second_distance}, "
                     f"rule={m.reason}, geometry error={m.geometric_error}, đúng={m.correct_geometry}.")
        else:index=None
        _plot(matching_view(a,b,ka,kb,matches,selected=index,presentation=_present()))
        st.caption("Ground-truth affine chỉ hậu kiểm, KHÔNG chọn match. Nearest, mutual cross-check và "
                   "ratio là ba quy tắc khác nhau; bit WTA_K=2, Hamming thông thường.")
    else:st.info("Thay đổi ảnh/tham số làm kết quả cũ hết hiệu lực. Bấm Calculate để tính lại.")
    st.write("SIFT float32: 128×4=512 byte; ORB-style: 256/8=32 byte → payload 16:1.")
    st.caption("Không benchmark Python teaching engine với OpenCV SIFT tối ưu; không suy ra tỉ lệ tốc độ cố định.")
    st.button("Reset Lab C",on_click=_reset_lab,args=("orb_c_",))


def render():
    _init()
    st.session_state.orb_plot_count=0
    st.markdown('<style>.orb-heading{margin:.2rem 0 .3rem;color:#123b5d;font-size:2.2rem;line-height:1.1}'
                '.orb-subtitle{color:#3d4b60;margin:.1rem 0 .55rem}'
                '@media(max-width:600px){.orb-heading{font-size:1.5rem;margin:0 0 .2rem}'
                '.orb-subtitle{font-size:.84rem;line-height:1.3}}</style>',unsafe_allow_html=True)
    if _present():st.markdown('<style>.block-container{max-width:1500px;padding-top:1.1rem}'
        '.orb-subtitle{display:none}.chapter-kicker{display:none}'
        '@media(max-width:600px){.orb-heading{display:none}}</style>',unsafe_allow_html=True)
    left,right=st.columns([2,1],vertical_alignment="bottom")
    with left:
        st.markdown('<div class="chapter-kicker">CHAPTER 2 · INTERACTIVE FEATURE LAB</div>',unsafe_allow_html=True)
        st.markdown('<h1 class="orb-heading">FAST · BRIEF · ORB-style</h1>',unsafe_allow_html=True)
        st.markdown('<p class="orb-subtitle">FAST-12 → pyramid → centroid → 256-bit oriented BRIEF → Hamming. '
                    'Teaching model, không byte-equivalent OpenCV ORB.</p>',unsafe_allow_html=True)
    with right:
        st.radio("Không gian học",("Guided Tutorial","Experiment Lab"),horizontal=True,key="orb_mode",on_change=_save_view)
        st.toggle("Presentation Mode",key="orb_presentation",on_change=_save_view)
    if st.session_state.orb_mode=="Guided Tutorial":_guided()
    else:
        tabs=st.tabs(("A · FAST","B · Orientation/BRIEF","C · Matching"))
        with tabs[0]:_lab_a()
        with tabs[1]:_lab_b()
        with tabs[2]:_lab_c()
