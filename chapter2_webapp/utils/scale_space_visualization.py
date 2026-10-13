"""Plotly views of computed scale-space arrays (no numerical decisions here)."""

import numpy as np
import plotly.graph_objects as go
from html import escape


def heatmap(values, title, *, signed=False, point=None, markers=None, presentation=False):
    a = np.asarray(values)
    bound = max(float(np.max(np.abs(a))), 1e-12) if signed else None
    fig = go.Figure(go.Heatmap(z=a, colorscale="RdBu_r" if signed else "Gray",
                               zmin=-bound if signed else 0,
                               zmax=bound if signed else 1,
                               colorbar=dict(title="signed" if signed else "I"),
                               hovertemplate="x=%{x}, y=%{y}<br>%{z:.6f}<extra></extra>"))
    if markers:
        fig.add_trace(go.Scatter(x=[m[1] for m in markers], y=[m[0] for m in markers],
                                 mode="markers", marker=dict(symbol="circle-open", size=15,
                                 color="#f97316", line=dict(width=3)), name="Ứng viên"))
    if point is not None:
        fig.add_trace(go.Scatter(x=[point[1]], y=[point[0]], mode="markers",
                                 marker=dict(symbol="cross", size=19, color="#ef4444",
                                             line=dict(width=3)), name="Pixel"))
    fig.update_layout(title=title, height=540 if presentation else 385,
                      margin=dict(l=35, r=25, t=55, b=45), paper_bgcolor="white",
                      font=dict(size=17 if presentation else 12, color="#172033"),
                      showlegend=bool(markers or point))
    fig.update_xaxes(title="x = cột", range=[-.5, a.shape[1]-.5], constrain="domain")
    fig.update_yaxes(title="y = hàng (xuống)", range=[a.shape[0]-.5, -.5],
                     scaleanchor="x", scaleratio=1)
    return fig


def curve(x, series, title, *, presentation=False):
    fig = go.Figure()
    for label, values in series.items():
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=label,
                                 marker=dict(size=10 if presentation else 7)))
    fig.update_layout(title=title, height=500 if presentation else 340,
                      xaxis_title="σ (pixel)", yaxis_title="Đáp ứng signed",
                      margin=dict(l=48, r=20, t=55, b=50),
                      font=dict(size=17 if presentation else 12), paper_bgcolor="white")
    return fig


def neighborhoods(stack, layer, row, col, current_event=None, presentation=False):
    """Three actual 3x3 arrays, same signed color scale; center is excluded in middle."""
    from plotly.subplots import make_subplots
    blocks = [stack[j][row-1:row+2, col-1:col+2] for j in (layer-1, layer, layer+1)]
    bound = max(max(float(np.max(np.abs(b))) for b in blocks), 1e-12)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("D thấp", "D hiện tại", "D cao"))
    for n, block in enumerate(blocks, start=1):
        labels = [[f"{v:+.4f}" for v in line] for line in block]
        fig.add_trace(go.Heatmap(z=block, text=labels, texttemplate="%{text}",
                                 textfont=dict(size=17 if presentation else 13),
                                 colorscale="RdBu_r", zmin=-bound, zmax=bound,
                                 showscale=n == 3,
                                 hovertemplate="%{z:.7f}<extra></extra>"), row=1, col=n)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{n if n > 1 else ''}",
                         scaleratio=1, row=1, col=n)
    if current_event is not None:
        col_idx = current_event.layer-layer+2
        fig.add_trace(go.Scatter(x=[current_event.col-col+1],
                                 y=[current_event.row-row+1], mode="markers",
                                 marker=dict(symbol="circle-open", color="#f97316", size=30,
                                             line=dict(width=4)), showlegend=False),
                      row=1, col=col_idx)
    fig.update_layout(height=400 if presentation else 300, margin=dict(l=15,r=15,t=52,b=10),
                      paper_bgcolor="white", font=dict(size=16 if presentation else 12))
    return fig


def neighborhood_html(stack, layer, row, col, current_event=None):
    """Responsive numeric 3x3x3 inspector: wraps vertically on narrow viewports."""
    panels = []
    for offset, label in ((-1,"Tầng thấp · 9"),(0,"Tầng hiện tại · 8"),(1,"Tầng cao · 9")):
        cells = []
        for rr in range(row-1,row+2):
            for cc in range(col-1,col+2):
                classes = "ss-cell"
                if offset == 0 and (rr,cc) == (row,col):
                    classes += " ss-center"
                if current_event is not None and (current_event.layer,current_event.row,current_event.col) == (layer+offset,rr,cc):
                    classes += " ss-compared"
                value = float(stack[layer+offset][rr,cc])
                cells.append(f'<div class="{classes}">{value:+.4f}</div>')
        panels.append(f'<div class="ss-panel"><strong>{escape(label)}</strong>'
                      f'<div class="ss-grid">{"".join(cells)}</div></div>')
    return ('<style>.ss-panels{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;}'
            '.ss-panel{flex:1 1 245px;min-width:0;background:#f7fafc;border:1px solid #d1dce5;'
            'border-radius:10px;padding:10px;color:#172033}.ss-grid{display:grid;'
            'grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;margin-top:8px}'
            '.ss-cell{background:#e5edf3;border-radius:4px;text-align:center;padding:10px 2px;'
            'font-variant-numeric:tabular-nums;font-size:clamp(11px,1.2vw,17px);font-weight:600}'
            '.ss-center{background:#fbd3b6;border:2px solid #d95822}'
            '.ss-compared{background:#fef08a;border:2px solid #ca8a04}'
            '@media(max-width:600px){.ss-cell{font-size:15px}.ss-panel{flex-basis:100%}}'
            '</style><div class="ss-panels">'+''.join(panels)+'</div>')
