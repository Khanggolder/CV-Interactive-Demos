"""Plotly views of SIFT engine arrays; no detector decisions are made here."""

import numpy as np
import plotly.graph_objects as go


COLORS = {"raw":"#e98224", "refined":"#1488aa", "kept":"#15936c",
          "low contrast":"#a45c20", "edge":"#d74449", "invalid":"#8067a8"}


def image_view(image, title, *, points=(), selected=None, signed=False, presentation=False):
    a = np.asarray(image)
    bound = max(float(np.max(np.abs(a))),1e-12) if signed else None
    fig = go.Figure(go.Heatmap(z=a, colorscale="RdBu_r" if signed else "Gray",
                               zmin=-bound if signed else 0,zmax=bound if signed else 1,
                               showscale=False,hovertemplate="x=%{x}, y=%{y}<br>%{z:.5f}<extra></extra>"))
    groups = {}
    for p in points:
        groups.setdefault(p.status,[]).append(p)
    for status,items in groups.items():
        fig.add_trace(go.Scatter(x=[p.col_global for p in items],y=[p.row_global for p in items],
                                 mode="markers",name=status,marker=dict(symbol="circle-open",
                                 size=17 if presentation else 12,color=COLORS.get(status,"#15936c"),
                                 line=dict(width=3))))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[selected.col_global],y=[selected.row_global],
                                 mode="markers",name="đang xem",marker=dict(symbol="cross",
                                 size=24 if presentation else 19,color="#e92637",line=dict(width=3))))
    fig.update_layout(title=title,height=555 if presentation else 390,
                      margin=dict(l=35,r=15,t=55,b=35),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12,color="#172033"),
                      legend=dict(orientation="h",y=-.1))
    fig.update_xaxes(title="x = cột",range=[-.5,a.shape[1]-.5])
    fig.update_yaxes(title="y = hàng (xuống)",range=[a.shape[0]-.5,-.5],scaleanchor="x",scaleratio=1)
    return fig


def response_patch(stack, kp, *, presentation=False):
    a = stack[kp.layer][kp.row-3:kp.row+4,kp.col-3:kp.col+4]
    fig = go.Figure(go.Heatmap(z=a,colorscale="RdBu_r",zmid=0,
                               text=np.array([[f"{v:+.4f}" for v in line] for line in a]),
                               texttemplate="%{text}",textfont=dict(size=20 if presentation else 16),
                               hovertemplate="DoG=%{z:+.7f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[3,3+kp.col_local-kp.col],y=[3,3+kp.row_local-kp.row],
                             mode="markers",name="lưới → refined",marker=dict(size=16,
                             color=["#e98224","#15936c"])))
    fig.update_layout(title="Lân cận DoG · lưới → subpixel",height=560 if presentation else 420,
                      margin=dict(l=25,r=10,t=55,b=20),font=dict(size=16 if presentation else 11),
                      paper_bgcolor="white")
    fig.update_xaxes(range=[-.5,6.5],domain=[.27,.73],showticklabels=False)
    fig.update_yaxes(range=[6.5,-.5],showticklabels=False)
    return fig


def gradient_arrows(image, kp, info, selected=None, *, presentation=False):
    """Real local gradient samples around the same keypoint; length reflects magnitude."""
    from utils.sift import gaussian_level
    a,j=gaussian_level(kp,image)
    radius=int(min(12,max(7,round(3*kp.sigma_local))))
    row,col=int(round(kp.row_local)),int(round(kp.col_local))
    y0,y1=max(0,row-radius),min(a.shape[0],row+radius+1)
    x0,x1=max(0,col-radius),min(a.shape[1],col+radius+1)
    fig=go.Figure(go.Heatmap(z=a[y0:y1,x0:x1],x=np.arange(x0,x1),y=np.arange(y0,y1),
                             colorscale="Gray",zmin=0,zmax=1,showscale=False))
    chosen=info.contributions[selected] if selected is not None and info.contributions else None
    for rr,cc,mag,angle,spatial,weight in info.contributions:
        if not (y0<=rr<y1 and x0<=cc<x1 and (rr-row)%3==0 and (cc-col)%3==0): continue
        length=min(1.4,mag*12)
        phi=np.radians(angle)
        color="#e98224" if chosen is not None and chosen[:2]==(rr,cc) else "#1685a8"
        fig.add_trace(go.Scatter(x=[cc,cc+length*np.cos(phi),None],
                                 y=[rr,rr+length*np.sin(phi),None],mode="lines",
                                 line=dict(color=color,width=3),showlegend=False,
                                 hovertemplate=f"(x={cc},y={rr}) |∇|={mag:.4f}, θ={angle:.1f}°, w={weight:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[kp.col_local],y=[kp.row_local],mode="markers",
                             marker=dict(size=17,color="#e3293a"),name="keypoint"))
    fig.update_layout(title=f"Gradient thật · Gaussian j={j} · y xuống",
                      height=520 if presentation else 390,margin=dict(l=25,r=10,t=55,b=20),
                      paper_bgcolor="white",font=dict(size=16 if presentation else 12))
    fig.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1)
    return fig


def oriented_patch_view(patch, *, presentation=False):
    fig=image_view(patch,"Patch theo frame của keypoint · lưới 4×4",presentation=presentation)
    side=patch.shape[0]
    fig.update_xaxes(range=[-.5,side-.5],domain=[.28,.72],showticklabels=False)
    fig.update_yaxes(range=[side-.5,-.5],showticklabels=False,scaleanchor=None)
    for position in np.linspace(-.5,side-.5,5):
        fig.add_vline(x=position,line_color="#e98224",line_width=2)
        fig.add_hline(y=position,line_color="#e98224",line_width=2)
    fig.update_layout(height=500 if presentation else 380)
    return fig


def orientation_histogram(info, selected_sample=None, *, presentation=False):
    x=np.arange(36)*10
    colors=["#16936b" if float(angle) in info.peaks else "#5985ae" for angle in x]
    fig=go.Figure(go.Bar(x=x,y=info.histogram,marker_color=colors,width=8,name="36 bin"))
    if selected_sample is not None and info.contributions:
        _,_,mag,angle,spatial,weight=info.contributions[selected_sample]
        fig.add_vline(x=angle,line_color="#e98224",line_dash="dash",
                      annotation_text=f"pixel: {angle:.1f}° · {weight:.4f}")
    fig.update_layout(title="Histogram hướng · Gaussian × gradient, làm mượt 5 bin",
                      height=480 if presentation else 340,xaxis_title="θ · độ (y ảnh hướng xuống)",
                      yaxis_title="trọng số",margin=dict(l=45,r=10,t=55,b=45),
                      paper_bgcolor="white",font=dict(size=17 if presentation else 12))
    fig.update_xaxes(range=[-5,355],dtick=30)
    return fig


def descriptor_grid(cells, selected=(0,0), *, presentation=False):
    sums=np.asarray(cells).sum(axis=2)
    fig=go.Figure(go.Heatmap(z=sums,colorscale="Blues",showscale=False,
                             text=np.array([[f"{v:.3f}" for v in row] for row in sums]),
                             texttemplate="%{text}",textfont=dict(size=20 if presentation else 15)))
    fig.add_trace(go.Scatter(x=[selected[1]],y=[selected[0]],mode="markers",
                             marker=dict(symbol="square-open",size=56 if presentation else 44,
                                         color="#e98224",line=dict(width=4)),showlegend=False))
    fig.update_layout(title="4×4 ô · tổng trọng số thực",height=480 if presentation else 340,
                      margin=dict(l=20,r=10,t=55,b=25),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12))
    fig.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1)
    return fig


def eight_bins(values, title, *, presentation=False):
    fig=go.Figure(go.Bar(x=np.arange(8)*45,y=values,width=34,marker_color="#3379ad"))
    fig.update_layout(title=title,height=430 if presentation else 305,
                      xaxis_title="góc tương đối (°)",yaxis_title="trọng số",
                      margin=dict(l=45,r=10,t=55,b=45),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12))
    return fig


def vector_view(values, title, *, presentation=False):
    a=np.asarray(values).reshape(16,8)
    fig=go.Figure(go.Heatmap(z=a,colorscale="Viridis",hovertemplate="ô=%{y}, bin=%{x}, d=%{z:.6f}<extra></extra>"))
    fig.update_layout(title=title,height=510 if presentation else 370,
                      xaxis_title="8 bin trong mỗi ô",yaxis_title="16 ô theo row-major",
                      margin=dict(l=55,r=15,t=55,b=40),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12))
    fig.update_yaxes(autorange="reversed")
    return fig


def matching_view(a,b,points_a,points_b,matches,selected=None,*,presentation=False):
    h,w=a.shape
    canvas=np.concatenate((a,b),axis=1)
    fig=go.Figure(go.Heatmap(z=canvas,colorscale="Gray",zmin=0,zmax=1,showscale=False))
    for m in matches:
        if m.nearest is None or m.query>=len(points_a) or m.nearest>=len(points_b): continue
        p,q=points_a[m.query],points_b[m.nearest]
        fig.add_trace(go.Scatter(x=[p.col_global,w+q.col_global],
                                 y=[p.row_global,q.row_global],mode="lines+markers",
                                 line=dict(color="#15936c" if m.accepted else "#d74449",width=3 if selected==m.query else 1),
                                 marker=dict(size=9 if selected==m.query else 5),showlegend=False,
                                 opacity=1 if selected==m.query else .35))
    fig.update_layout(title="Ảnh A → B · xanh: ratio pass, đỏ: reject",height=600 if presentation else 420,
                      margin=dict(l=20,r=15,t=55,b=25),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12))
    fig.update_xaxes(range=[-.5,2*w-.5]);fig.update_yaxes(range=[h-.5,-.5],scaleanchor="x",scaleratio=1)
    return fig
