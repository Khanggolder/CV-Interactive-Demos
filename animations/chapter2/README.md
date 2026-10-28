# Chapter 2 Manim · Harris and SIFT

Two lecture animations extend the Chapter 2 Guided Tutorials. They use the
Streamlit-free numerical engines in `chapter2_webapp/utils/`; neither animation
imports a Streamlit UI module. Both are self-contained in this directory apart
from those existing engines and the root dependencies.

| Scene | Source | Storyboard | Target |
| --- | --- | --- | --- |
| `HarrisCornerDetection` | `harris_corner.py` | flat/edge/corner → shift energy → Sobel and Gaussian-integrated tensor → eigenvalues → signed Harris response → threshold and 5×5 NMS → detected corner and scale limitation | 50–65 s |
| `SIFTFeaturePipeline` | `sift_pipeline.py` | transformed view → Gaussian octaves → signed DoG → strict 26-neighbor candidate → Taylor refinement → contrast/edge filters → 36-bin orientation → rotated local frame → 4×4×8 descriptor → L2/clip/L2 → Lowe ratio and post-hoc affine check → overview | 80–100 s |

The visual system follows Chapter 3: 16:9, navy `#0B1020`, blue image data,
orange active responses, yellow inspection, green accepted results, purple
scale/orientation, and muted gray inactive elements. Captions change as whole
objects. The source files contain no LaTeX dependency.

## Numerical conventions and fixtures

- Arrays are grayscale float `[0,1]` indexed `[row,col]`; displayed image x
  points right and image y points down. Display markers are mapped from the
  actual row/column coordinates.
- Harris uses the engine's 41×41 flat, horizontal-edge, and L-corner presets.
  The selected L-corner is at `[20,20]`; the 9×9 unpadded window shift energy,
  Sobel correlation kernels divided by 8, 5×5 normalized Gaussian tensor
  integration (`σ=1.2`), `k=.05`, and threshold `0.08 × max(R)` are computed
  from `utils/harris.py`. The 5×5 NMS mask retains `[20,20]`. Equal maxima use
  the engine's top-left tie rule. Harris's fixed scale gives no scale guarantee.
  Displacement arrows move by exactly one displayed source pixel per unit of
  `(u,v)`. The displayed 3×3 difference/square crop comes from the same real
  9×9 comparison; the energy sum uses all 81 terms.
- SIFT starts with `"Blob và góc rõ"` (129×129). Its actual pyramid uses
  `σ_base=1.6`, three intervals and three octaves; `σ_global=σ_local·2^octave`.
  The displayed DoG is `L_{j+1}−L_j`, with blue negative and orange positive.
  Its 3×3×3 neighborhood is shown with values multiplied by 1000 solely to
  make the 26 individual comparisons readable.
  For this fixture there are 12 raw DoG candidates, 7 kept locations and 23
  orientation records; these are distinct quantities. The selected octave-0
  candidate moves its integer grid site during Taylor refinement. The final
  Taylor displacement is measured at the refined grid site. A separate
  deterministic `"Cảnh nhiễu"` fixture supplies a genuine low-contrast and
  edge-like rejection example, explicitly labeled in the animation.
- Matching uses the asymmetric deterministic preset and a known 20°,
  scale-0.92 affine warp. The educational engine computes all descriptors and
  Lowe decisions first. The affine matrix is consulted only afterwards to
  display one accepted geometrically accurate match and one accepted but
  inaccurate match. The latter demonstrates why descriptor acceptance is not
  a geometric proof. The descriptor has 128 actual components, ordered
  `(cell_row*4+cell_col)*8+orientation_bin`. The sample's cell/bin highlight
  comes from the engine's trilinear descriptor sampling.
- Signed response colors and enlarged image tiles are display mappings only.
  The numerical arrays, histogram, matrices, distances, decisions and markers
  remain unchanged. The 4×4 cell grid is an abstract spatial layout; its
  support width depends on the selected `σ_local`, not a universal 16×16 patch.
  This teaching implementation is not bit equivalent to production OpenCV SIFT.

## Reproduce on Windows PowerShell

Use the existing root `.venv` and root `requirements.txt`. No additional
environment, system LaTeX or external image dataset is needed.

```powershell
Push-Location animations\chapter2
..\..\.venv\Scripts\python.exe -B verify_harris.py --preflight
..\..\.venv\Scripts\python.exe -B verify_sift.py --preflight
..\..\.venv\Scripts\python.exe -B -m manim -ql --disable_caching -o HarrisCornerDetection_review_v6.mp4 harris_corner.py HarrisCornerDetection
..\..\.venv\Scripts\python.exe -B -m manim -ql --disable_caching -o SIFTFeaturePipeline_review_v5.mp4 sift_pipeline.py SIFTFeaturePipeline
..\..\.venv\Scripts\python.exe -B verify_harris.py --video review --contact-sheet
..\..\.venv\Scripts\python.exe -B verify_sift.py --video review --contact-sheet
..\..\.venv\Scripts\python.exe -B -m manim -qh --disable_caching -o HarrisCornerDetection_final_v4.mp4 harris_corner.py HarrisCornerDetection
..\..\.venv\Scripts\python.exe -B -m manim -qh --disable_caching -o SIFTFeaturePipeline_final_v4.mp4 sift_pipeline.py SIFTFeaturePipeline
..\..\.venv\Scripts\python.exe -B verify_harris.py --video final
..\..\.venv\Scripts\python.exe -B verify_sift.py --video final
Pop-Location
```

The verifiers independently recompute displayed arithmetic and decision
rules. `--preflight` exercises every scene transition with finite 16:9 bounds
checks and confirms no updaters remain. `--video` opens the MP4, decodes every
frame, checks resolution, frame rate, duration, nonempty output and long blank
intervals. The contact sheets supplement manual review of rendered frames.

Generated files stay under Git-ignored `media/`:

- Reviews: `media/videos/harris_corner/480p15/HarrisCornerDetection_review_v6.mp4`
  and `media/videos/sift_pipeline/480p15/SIFTFeaturePipeline_review_v5.mp4`.
- Finals: `media/videos/harris_corner/1080p60/HarrisCornerDetection_final_v4.mp4`
  and `media/videos/sift_pipeline/1080p60/SIFTFeaturePipeline_final_v4.mp4`.
- QA: `media/harris_review_contact.png`, `media/sift_review_contact.png`.

The Guided Tutorials remain interactive and allow changes to fixtures and
parameters; these videos follow one fixed, verified path for lectures.
