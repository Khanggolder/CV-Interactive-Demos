"""Plotly figures built only from computed ORB teaching data."""

import numpy as np
import plotly.graph_objects as go

from utils.orb import CIRCLE, CARDINAL

INK="#172033"
COLORS={"BRIGHT":"#158063","DARK":"#c54846","SIMILAR":"#52657b"}


def _layout(fig,title,presentation=False,height=420):
    fig.update_layout(title=title,height=height+(100 if presentation else 0),
                      margin=dict(l=35,r=20,t=60,b=35),paper_bgcolor="white",
                      font=dict(size=17 if presentation else 13,color=INK),showlegend=False)
    return fig


def fast_circle(image,row,col,result,*,shown=16,show_arc=False,presentation=False):
    patch=image[row-3:row+4,col-3:col+4]
    fig=go.Figure(go.Heatmap(z=patch,colorscale="Greys",zmin=0,zmax=255,
                             showscale=False,hovertemplate="x=%{x}, y=%{y}<br>I=%{z:.1f}<extra></extra>"))
    def cell(x,y,color,label,cardinal=False):
        fig.add_shape(type="rect",x0=x-.47,x1=x+.47,y0=y-.47,y1=y+.47,
                      fillcolor=color,line=dict(color="#123b5d" if cardinal else "white",
                                                width=3 if cardinal else 1.3),layer="above")
        fig.add_annotation(x=x,y=y,text=label,showarrow=False,
                           font=dict(size=14 if presentation else 12,color="white"),
                           align="center",xanchor="center",yanchor="middle")
    cell(3,3,"#123b5d",f"p<br>{result.center:.0f}")
    for i,(dr,dc) in enumerate(CIRCLE):
        if i>=shown:continue
        active=show_arc and i in result.arc
        cell(3+dc,3+dr,"#e78016" if active else COLORS[result.labels[i]],
             f"{i+1}<br>{result.values[i]:.0f}",i in CARDINAL)
    _layout(fig,"FAST · vòng 16 pixel",presentation,430)
    fig.update_xaxes(range=[-.6,6.6],dtick=1,showticklabels=False,constrain="domain")
    fig.update_yaxes(range=[6.6,-.6],dtick=1,showticklabels=False,
                     scaleanchor="x",scaleratio=1,constrain="domain")
    return fig


def image_view(image,title,*,points=(),selected=None,presentation=False):
    a=np.asarray(image)
    fig=go.Figure(go.Heatmap(z=a,colorscale="Greys",zmin=0,zmax=255,showscale=False,
                             hovertemplate="x=%{x}, y=%{y}<br>I=%{z:.1f}<extra></extra>"))
    if points:
        fig.add_trace(go.Scatter(x=[p.col for p in points],y=[p.row for p in points],
            mode="markers",marker=dict(symbol="circle-open",size=14 if presentation else 10,
            color="#0c996f",line=dict(width=2))))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[selected[1]],y=[selected[0]],mode="markers",
            marker=dict(symbol="cross",size=25,color="#e56917",line=dict(width=3))))
    _layout(fig,title,presentation,420)
    fig.update_xaxes(title="x = cột",range=[-.5,a.shape[1]-.5],constrain="domain")
    fig.update_yaxes(title="y = hàng (xuống)",range=[a.shape[0]-.5,-.5],
                     scaleanchor="x",scaleratio=1,constrain="domain")
    return fig


def score_view(scores,eligible,nms,selected=None,presentation=False):
    fig=go.Figure(go.Heatmap(z=scores,colorscale="YlOrRd",showscale=True,
                             hovertemplate="x=%{x}, y=%{y}<br>FAST score=%{z:.2f}<extra></extra>"))
    rr,cc=np.nonzero(nms)
    fig.add_trace(go.Scatter(x=cc,y=rr,mode="markers",marker=dict(symbol="circle-open",
        size=15 if presentation else 10,color="#127a62",line=dict(width=2))))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[selected[1]],y=[selected[0]],mode="markers",
                                 marker=dict(symbol="x",size=18,color="#123b5d")))
    _layout(fig,"FAST margin score · xanh = NMS giữ",presentation,420)
    fig.update_xaxes(range=[-.5,scores.shape[1]-.5],constrain="domain")
    fig.update_yaxes(range=[scores.shape[0]-.5,-.5],scaleanchor="x",scaleratio=1,
                     constrain="domain")
    return fig


def centroid_view(moments,*,selected=None,presentation=False):
    a=np.where(moments.mask,moments.patch,np.nan)
    rad=a.shape[0]//2
    fig=go.Figure(go.Heatmap(z=a,x=np.arange(-rad,rad+1),y=np.arange(-rad,rad+1),
                             colorscale="Greys",zmin=0,zmax=255,showscale=False,
                             hovertemplate="x=%{x}, y=%{y}<br>I=%{z:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0],y=[0],mode="markers",marker=dict(size=17,color="#0b5688")))
    if moments.cx is not None:
        fig.add_trace(go.Scatter(x=[0,moments.cx],y=[0,moments.cy],mode="lines+markers",
                                 line=dict(color="#e66b13",width=5),marker=dict(size=17,color="#e66b13")))
    if selected is not None:
        x,y=selected
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers",marker=dict(symbol="square-open",size=29,
                                      color="#d62f43",line=dict(width=3))))
    _layout(fig,"Moment cường độ · O → centroid C",presentation,460)
    fig.update_xaxes(title="Δx = cột",dtick=3,range=[-rad-.5,rad+.5],domain=[.29,.71])
    fig.update_yaxes(title="Δy = hàng (xuống)",autorange="reversed",scaleanchor="x",scaleratio=1,dtick=3)
    return fig


def brief_pair_view(image,row,col,sample,*,presentation=False):
    y0,y1=max(0,row-10),min(image.shape[0],row+11)
    x0,x1=max(0,col-10),min(image.shape[1],col+11)
    patch=image[y0:y1,x0:x1]
    fig=go.Figure(go.Heatmap(z=patch,colorscale="Greys",zmin=0,zmax=255,showscale=False))
    px,py,qx,qy,ip,iq=sample
    fig.add_trace(go.Scatter(x=[px-x0,qx-x0],y=[py-y0,qy-y0],
        mode="lines+markers+text",text=[f"p:{ip:.0f}",f"q:{iq:.0f}"],
        textposition=["top left","bottom right"],textfont=dict(size=18 if presentation else 14,color="#123b5d"),
        line=dict(width=4,color="#e56917"),marker=dict(size=18,color=["#1c74a6","#16a075"])))
    _layout(fig,"BRIEF · hai pixel được lấy mẫu sau Gaussian",presentation,470)
    fig.update_xaxes(range=[-.5,patch.shape[1]-.5],showticklabels=False,domain=[.18,.82])
    fig.update_yaxes(range=[patch.shape[0]-.5,-.5],showticklabels=False,scaleanchor="x",scaleratio=1)
    return fig


def bits_view(bits,*,selected=0,presentation=False):
    a=np.asarray(bits,dtype=int).reshape(-1)
    start=max(0,min((selected//32)*32,max(0,len(a)-32)))
    shown=a[start:start+32].reshape(4,8)
    fig=go.Figure(go.Heatmap(z=shown,zmin=0,zmax=1,colorscale=[[0,"#dce8ef"],[1,"#158063"]],
        showscale=False,text=shown,texttemplate="%{text}",textfont=dict(size=22 if presentation else 17),
        hovertemplate="bit=%{z}<extra></extra>"))
    _layout(fig,f"Descriptor bits {start}–{start+31} · 4 bytes (MSB trước)",presentation,280)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(autorange="reversed",showticklabels=False,scaleanchor="x",scaleratio=1)
    return fig


def matching_view(a,b,points_a,points_b,matches,*,selected=None,presentation=False,
                  reveal=True):
    h,w=a.shape
    fig=go.Figure(go.Heatmap(z=np.concatenate((a,b),axis=1),colorscale="Greys",zmin=0,zmax=255,
                             showscale=False))
    shown=[m for m in matches if m.train is not None and (not reveal or m.accepted)]
    for match in shown[:30]:
        p,q=points_a[match.query],points_b[match.train]
        color=("#61758d" if not reveal else
               "#168262" if match.correct_geometry is True else "#d14a4a")
        fig.add_trace(go.Scatter(x=[p.x_original,w+q.x_original],y=[p.y_original,q.y_original],
            mode="lines+markers",line=dict(color=color,width=4 if match.query==selected else 1.6),
            marker=dict(size=10 if match.query==selected else 5),opacity=1 if match.query==selected else .6,
            hovertemplate=(f"Hamming {match.distance}; geometry error {match.geometric_error:.1f}px<extra></extra>"
                           if reveal else f"Hamming {match.distance}<extra></extra>")))
    title=("A → B · xanh = đúng hình học; đỏ = sai (chỉ các match được chấp nhận)"
           if reveal else "A → B · ứng viên trung tính, chưa reveal quyết định/hình học")
    _layout(fig,title,presentation,520)
    fig.update_xaxes(range=[-.5,2*w-.5],constrain="domain")
    fig.update_yaxes(range=[h-.5,-.5],scaleanchor="x",scaleratio=1,constrain="domain")
    return fig
