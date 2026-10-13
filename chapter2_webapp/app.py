"""Chapter 2 learning lab: four feature tutorials and an applied matching challenge."""

from __future__ import annotations

import streamlit as st

from tutorials.harris import render as render_harris
from tutorials.scale_space import render as render_scale_space
from tutorials.sift import render as render_sift
from tutorials.orb import render as render_orb
from tutorials.feature_challenge import render as render_feature_challenge

st.set_page_config(page_title="Chapter 2 · Feature Lab", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  :root {--ink:#172033; --navy:#123b5d; --accent:#d95822;}
  .block-container {max-width:1380px; padding-top:2.7rem; padding-bottom:3rem;}
  h1,h2,h3 {color:var(--navy)!important;}
  p,li,label {color:var(--ink)!important;}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *
    {color:#334155!important; opacity:1!important;}
  [data-testid="stPlotlyChart"] {border:1px solid #d1dce5; border-radius:12px; background:white;}
  .chapter-kicker {color:var(--accent); font-weight:800; letter-spacing:.12em; font-size:.77rem;}
  .question {background:#fff2e9; border-left:5px solid var(--accent); padding:.8rem 1rem;
             color:#713515; border-radius:7px; font-weight:600;}
  .explain {background:#e8f6f0; border-left:5px solid #158063; padding:.8rem 1rem;
            color:#174a3e; border-radius:7px;}
  .chapter2-steps {display:flex; gap:4px; margin:.35rem 0 1rem;}
  .chapter2-steps span {height:7px; flex:1; border-radius:10px; background:#dce6ec;}
  .chapter2-steps span.done {background:#14966f;}
  .chapter2-steps span.active {background:var(--accent);}
  .presentation [data-testid="stPlotlyChart"] {max-width:none;}
  .st-key-feature_comparison_mobile {display:none;}
  .feature-result-card {border:1px solid #bdcfdb; border-radius:10px; background:#f7fafc;
    padding:.65rem .8rem; margin:.35rem 0; color:#172033; line-height:1.6;}
  .feature-result-card strong,.feature-result-card small {display:block;}
  .feature-result-card strong {color:#123b5d;font-size:1.05rem;}
  @media(max-width:900px){[data-testid="stHeader"]{display:none!important;}
    .block-container{padding-top:1.4rem;} h1{font-size:1.9rem!important;}
    .chapter-kicker{display:none;}}
  @media(max-width:600px){[data-testid="stHeader"]{display:none!important;}
    .block-container{padding-top:.9rem;padding-left:.8rem;padding-right:.8rem;}
    h1{font-size:1.55rem!important;} .chapter2-steps{gap:2px;}
    .feature-challenge-intro{display:none;}
    .st-key-feature_comparison_desktop {display:none!important;}
    .st-key-feature_comparison_mobile {display:block;}}
</style>
""", unsafe_allow_html=True)
module = st.selectbox("Bài học Chapter 2",
                      ("Harris & Shi-Tomasi Corner Detection",
                       "Gaussian Scale Space, LoG & DoG",
                       "SIFT Interactive Tutorial",
                       "FAST, BRIEF & ORB",
                       "Feature Matching Challenge"),
                      key="chapter2_module", label_visibility="collapsed")
if module == "Harris & Shi-Tomasi Corner Detection":
    render_harris()
elif module == "Gaussian Scale Space, LoG & DoG":
    render_scale_space()
elif module == "SIFT Interactive Tutorial":
    render_sift()
elif module == "FAST, BRIEF & ORB":
    render_orb()
else:
    render_feature_challenge()
