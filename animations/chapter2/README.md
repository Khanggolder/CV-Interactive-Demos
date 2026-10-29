# Chapter 2 Manim · Harris and SIFT

Four lecture animations extend the Chapter 2 Guided Tutorials. They use the
Streamlit-free numerical engines in `chapter2_webapp/utils/`; no animation
imports a Streamlit UI module. Their scene code and fixtures live in this
directory, with the existing engines and root dependencies.

| Scene | Source | Role | Target |
| --- | --- | --- | --- |
| `HarrisCornerDetection` | `harris_corner.py` | Approved Harris V1 | 50–65 s |
| `SIFTFeaturePipeline` | `sift_pipeline.py` | Short SIFT V1 overview | 80–100 s |
| `SIFTDeepDive` | `sift_deep_dive.py` | Original detailed V2, retained for reference | 7–9 min |
| `SIFTDeepDiveV21` | `sift_deep_dive_v21.py` | Recommended detailed V2.1 Final Polished | ~10.4 min |

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

## SIFT Deep Dive V2

`SIFTDeepDive` in `sift_deep_dive.py` is a separate 7–9 minute lesson. It does
not replace `SIFTFeaturePipeline`. `sift_deep_dive_data.py` builds one
deterministic 129×129 grayscale camera with an asymmetric body, concentric
lens, viewfinder, flash and small interior marks. The same image produces the
Gaussian pyramid, signed DoG, extrema, refined keypoints, orientations and
descriptors. Matching uses a 17° rotation and 0.94 scale copy of that camera.

The engine computes three octaves at 129², 65² and 33² pixels, with six
Gaussian and five DoG levels each: 18 Gaussian images and 15 signed differences.
The source image yields 104 strict DoG candidates, 68 successfully refined
candidates, 25 filtered keypoint locations and 42 orientation/descriptor
records. The transformed copy yields 31 descriptor records. One accepted Lowe
match has geometric error about 0.0104 px; another accepted match on the same
object has about 9.77 px error, so the lesson shows geometric verification as
a distinct post-match check. The known affine transform never chooses matches.

The video follows a measured camera pixel and the engine's 15×15 normalized
Gaussian kernel, incremental sigma, sampled octave transitions, parent-child
DoG subtraction, strict 26-neighbor comparisons, 3×3 Taylor localization,
an observed recenter from column 103 to 102 before the subpixel offset,
separate 2×2 edge rejection, 36-bin orientation voting, 4×4×8 descriptor
interpolation, L2/clip/L2 and Lowe ratio. A second valid orientation is shown
at one physical location. Lower-resolution octave arrays are enlarged with nearest
neighbor sampling only for the continuous Manim image transform; numerical
arrays and resolution labels retain their original values. This educational
engine is a discrete SIFT approximation rather than OpenCV bit equivalence.

From `animations/chapter2/` with the existing root `.venv`:

```powershell
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive.py --preflight
..\..\.venv\Scripts\python.exe -X utf8 -B -m manim -ql --disable_caching -o SIFTDeepDive_review.mp4 sift_deep_dive.py SIFTDeepDive
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive.py --video review --contact-sheet
..\..\.venv\Scripts\python.exe -X utf8 -B -m manim -qh --fps 30 --disable_caching -o SIFTDeepDive_final.mp4 sift_deep_dive.py SIFTDeepDive
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive.py --video final
```

Review: `media/videos/sift_deep_dive/480p15/SIFTDeepDive_review.mp4`.
Final: `media/videos/sift_deep_dive/1080p30/SIFTDeepDive_final.mp4`.
Contact sheet: `media/sift_deep_dive_contact.png`. All are ignored by Git.

## SIFT Deep Dive V2.1 — Final Polished (recommended)

`SIFTDeepDiveV21` in `sift_deep_dive_v21.py` reuses the V2 camera fixture and
numerical engines without changing the approved V2 scene or its MP4s. This is
the recommended detailed lecture animation; V2 remains available for reference.
It adds a fixed grayscale Gaussian pixel scan, an explicit `G(0,3)[::2,::2]` octave
transition, five labeled signed DoG responses for one pixel, and a continuous
transition from the selected D1 response to its three 3×3 neighbor matrices.
The D1 value is first identified as a minimum along scale at one pixel; only
the later strict 26-neighbor check identifies a raw 3D DoG candidate.

The orientation lesson computes central differences on the actual Gaussian
level, reverses image y only when drawing screen arrows, weights gradient
votes by distance, fills 36 initially empty bins, applies the engine's
five-tap circular smoothing, and selects the actual histogram peak. The
descriptor lesson follows the real oriented patch, splits one gradient among
up to eight 4×4×8 components, accumulates several sample contributions into
one component, builds all 16 cell histograms, and traces their indices into
128D. The final stages show L2/clip/L2, Euclidean descriptor distance, Lowe
ratio, and the separately revealed affine geometry check.

Run from `animations/chapter2/` with the existing root `.venv`:

```powershell
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive_v21.py --preflight
..\..\.venv\Scripts\python.exe -X utf8 -B -m manim -ql --disable_caching -o SIFTDeepDive_V21_review_polished.mp4 sift_deep_dive_v21.py SIFTDeepDiveV21
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive_v21.py --video review --contact-sheet
..\..\.venv\Scripts\python.exe -X utf8 -B -m manim -qh --fps 30 --disable_caching -o SIFTDeepDive_V21_final_polished.mp4 sift_deep_dive_v21.py SIFTDeepDiveV21
..\..\.venv\Scripts\python.exe -X utf8 -B verify_sift_deep_dive_v21.py --video final
```

The verifier is `verify_sift_deep_dive_v21.py`; it also runs the V2 checks in
`verify_sift_deep_dive.py`. Both scenes depend on `sift_deep_dive_data.py`,
`chapter2_common.py`, and the tracked `chapter2_webapp/utils/scale_space.py`
and `sift.py` engines. The root `requirements.txt` supplies Python packages.

Recommended local final:
`media/videos/sift_deep_dive_v21/1080p30/SIFTDeepDive_V21_final_polished.mp4`.
Local review:
`media/videos/sift_deep_dive_v21/480p15/SIFTDeepDive_V21_review_polished.mp4`.
These MP4s are Git-ignored local outputs, not hosted in this repository. The
V2 final under `media/videos/sift_deep_dive/` and the original V2.1 final are
retained as historical versions. The polished V2.1's Gaussian panel labels
the 3×3 kernel excerpt within the full 15×15 kernel;
the trilinear panel shows magnitude, Gaussian weight and all four factors
before summarizing the eight computed contributions.
