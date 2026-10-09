# Chapter 3 Manim animations

Teaching animations for Chapter 3: Canny, point-to-curve Hough Transform,
Hough accumulator voting, and RANSAC line fitting.

## Scene

- `CannyPipelinePrototype`: visualizes the six stages from the original image to the final edge map.
- The input is generated deterministically in NumPy; no external image asset is required.
- The scene uses Manim `Text` instead of `Tex`/`MathTex`, so LaTeX is optional.
- A persistent 7×7 magnifier follows the same pixels through smoothing,
  gradient magnitude, and NMS. NMS compares actual q1/p/q2 samples for KEEP
  and DELETE; suppressed pixels darken in place.
- Hysteresis animates breadth-first propagation through the computed
  8-connected weak pixels. Gold = strong, orange = weak, white = accepted,
  muted gray = rejected/noise. Two real image crops show KEEP and REJECT.
- Numerical Canny functions are unchanged. The existing NMS uses four
  quantized directions; this is a teaching prototype, not subpixel NMS.

## Setup

From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r animations\chapter3\requirements.txt
```

## Render

Low-quality review render:

```powershell
cd animations\chapter3
..\..\.venv\Scripts\python -m manim -ql --disable_caching -o CannyPipelinePrototype_review.mp4 canny_pipeline.py CannyPipelinePrototype
..\..\.venv\Scripts\python verify_review.py
```

Final 1080p render after approval:

```powershell
cd animations\chapter3
..\..\.venv\Scripts\python -m manim -qh canny_pipeline.py CannyPipelinePrototype
```

Rendered media is written below `animations/chapter3/media/` and is ignored by Git.
The review video is `media/videos/canny_pipeline/480p15/CannyPipelinePrototype_review.mp4`.
`verify_review.py` opens and decodes every frame, checks the review resolution
and 24–30 second duration; `--contact-sheet` also generates a visual QA sheet.
Use `--disable_caching` when changing computed image data to avoid stale frames.

## Hough: point to sinusoid

`hough_transform.py` contains one scene, `HoughTransformPointToSinusoid`.
It follows Chapter 3's normal form `rho = x*cos(theta) + y*sin(theta)`.
Image coordinates use Cartesian y-up axes. Theta is the normal's angle in
`[0, pi)` and rho is signed; it can become negative during the sweep.

The fixed points `(1, 2)` and `(3, 1)` define `x + 2y = 5`. Their numerically
generated Hough curves meet at `rho = sqrt(5)`, `theta = atan2(2, 1)`.
Four more exactly collinear points illustrate the ideal shared location.
There is no discretized accumulator or voting grid in this scene.

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_hough.py
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o HoughTransformPointToSinusoid_review.mp4 hough_transform.py HoughTransformPointToSinusoid
..\..\.venv\Scripts\python -B verify_hough.py --video
```

The review is `media/videos/hough_transform/480p15/HoughTransformPointToSinusoid_review.mp4`.
`verify_hough.py` checks line membership, normals, finite geometry and numerical
curve intersections. `--video` decodes every frame and checks 480p and a
30–38 second duration; `--contact-sheet` also writes an optional QA image.
The scene checks text bounds after each transition and clears all updaters.

Blue links image points to their curves, yellow marks the active point/normal,
orange identifies the second point, and green links the detected line to the
parameter-space intersection. Hough text caches and QA output are scoped to
`media/hough_transform/`, separate from the approved Canny scene.

## Hough accumulator voting

`hough_accumulator.py` adds one continuous scene, `HoughAccumulatorVoting`.
It imports coordinate/style helpers from `hough_transform.py` without modifying
either approved scene. The image plane and Hough axes remain visible as the
continuous curves become a discretized accumulator.

- Theta bins: 18 bins of 10 degrees; samples at 5, 15, ..., 175 degrees.
- Rho bins: 20 half-open bins of width 0.5, covering [-5, 5).
- Each edge point contributes exactly 18 computed votes. Three enlarged cells
  below the grid display actual integer counts; their labels use zero-based
  `[rho row, theta column]` indices. No votes are clipped or invented.
- The six main points are unchanged from the previous Hough scene. Three
  distractors `(0.5, 0.3)`, `(2.5, 3.3)`, `(4.1, 2.6)` contribute scattered votes.
- `A[14,6]` is the unique maximum: 6 votes, versus a runner-up of 3. It covers
  rho in [2.0, 2.5) and theta in [60, 70) degrees.
- The displayed detected line is **the bin-center approximation**:
  `x*cos(65°) + y*sin(65°) = 2.25`. It is not silently replaced with the exact
  source line `x + 2y = 5`; the maximum support-point distance is about 0.122
  coordinate units. The coarse grid is chosen for visibility at 480p.

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_accumulator.py
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o HoughAccumulatorVoting_review.mp4 hough_accumulator.py HoughAccumulatorVoting
..\..\.venv\Scripts\python -B verify_accumulator.py --video
```

Review video: `media/videos/hough_accumulator/480p15/HoughAccumulatorVoting_review.mp4`.
`verify_accumulator.py` independently checks the accumulator with NumPy's 2D
histogram, the unique peak, bin/line consistency and finite geometry. `--video`
decodes all frames and checks the 25–35 second target; `--contact-sheet` adds
an optional QA image. Caches/logs are isolated in `media/hough_accumulator/`.
Counts are calculated from the sampled-angle prefix, independent of frame rate.
Blue marks data, yellow marks the active voter/high counts, and green marks the
final peak and detected line. This scene stops before probabilistic Hough,
line segments or any subsequent algorithm.

## RANSAC: sample, consensus, refit

`ransac.py` contains one scene, `RANSACLineFitting`. The same scatter plot stays
alive through ordinary least squares, three sampled hypotheses, and a final
total-least-squares (TLS/PCA) refit on the winning consensus set.

- Dataset seed `37`: 14 points with equally spaced x in `[-3.4, 3.4]`,
  `y = 0.65*x + 0.3 + N(0, 0.095^2)`, plus six fixed outliers. The complete
  numerical coordinates are printed by the verifier.
- Sample seed `7647`: the first three draws, without replacement within each
  pair, are `[14, 6]`, `[5, 2]`, `[2, 11]` (zero-based indices). Seeds are chosen
  for a teaching example; the scene does not claim three trials always suffice.
- The candidate is a normalized implicit line `a*x + b*y + c = 0`. Distances
  are perpendicular; `d <= 0.22` gives consensus counts `3`, `7`, and `14`.
- The epsilon band is the actual strip `|a*x + b*y + c| <= 0.22`, clipped to
  the plot. Equal x/y display scales preserve its geometric meaning. Right-angle
  helper marks distinguish perpendicular distances from vertical residuals.
- Point colors indicate membership in the **current candidate's** consensus,
  not privileged knowledge of the synthetic labels. Yellow marks a sampled pair;
  cyan marks accepted points; orange marks rejection; green marks the final fit.
- The introduction fits ordinary vertical-residual least squares to the first
  18 points, then recomputes on all 20 as two strong outliers are added. The final
  comparison uses that same all-data LS fit as a faint dashed line.
- TLS refits all 14 winning inliers: approximately `y = 0.657463*x + 0.299273`.
  The slight visible shift reduces RMS perpendicular residual from `0.116854`
  (two-point hypothesis) to `0.095022` (refit). No true labels enter the fit.

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_ransac.py --report
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o RANSACLineFitting_review.mp4 ransac.py RANSACLineFitting
..\..\.venv\Scripts\python -B verify_ransac.py --video --report
```

Review video: `media/videos/ransac/480p15/RANSACLineFitting_review.mp4`.
The verifier independently checks cross-product distances, TLS via covariance
eigenvectors, exact band width, consensus scores, finite geometry, and point
bounds. The scene checks text bounds after every animation. `--video` decodes
every frame and checks 480p/15fps and 30–40 seconds; `--contact-sheet` creates
an optional QA montage. `--report` saves full coordinates and metrics to
`media/ransac/validation.json`. Caches and logs are isolated in `media/ransac/`.
No prior scene is imported or modified. No iteration-count formula or other
Chapter 3 algorithm is introduced.

## LSD teaching prototype

`lsd.py` provides `LSDLineSegmentDetector`: a persistent 12×10 intensity grid,
computed finite-difference gradients, perpendicular level lines, strongest-seed
8-connected BFS, weighted-PCA support rectangles/segments, and binomial NFA.
Natural-language transitions use sequential whole-text fades, not glyph morphs.

The deterministic field uses seed 9. Growth compares axial orientations modulo
pi against the seed with tolerance 22.5 degrees; NFA alignment retains gradient
polarity modulo 2*pi, consistent with the requested `p0 = tau/pi = 1/8`.
All 20 grown regions count toward `N_T`; the first two are shown. Candidate A
has n=k=33 and NFA=3.15544e-29; B has n=4, k=2 and NFA=1.57715.
The accept/reject labels come from comparing computed NFA against 1.

This is a **teaching prototype**, not full reference LSD. It uses the slide's
LSR-based counts and the number of adaptively grown regions, not the reference
rectangle-test family/refinement. Consequently its displayed NFA is not a
calibrated whole-image false-alarm guarantee. Two candidates are animated;
the full computed region list is retained in the numerical report.
Reference: https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_lsd.py
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o LSDLineSegmentDetector_review.mp4 lsd.py LSDLineSegmentDetector
..\..\.venv\Scripts\python -B verify_lsd.py --video
```

Review: `media/videos/lsd/480p15/LSDLineSegmentDetector_review.mp4`.
Report: `media/lsd/validation.json` (full field, gradients, all BFS events,
regions, fitted parameters, NFA values and limitations). `--qa` also creates
a temporary contact sheet. All LSD caches/logs are scoped to `media/lsd/`.

## Sobel: pixels, products, gradient

`sobel_convolution.py` contains `SobelConvolution`. A persistent synthetic image
zooms into a 3×3 patch; the displayed masks overlay those same pixels. The first
three products are explained individually, the remaining six accelerate, and
row contributions collect into the total. The mask changes from Gx to Gy without
rebuilding the image. The measured components construct the gradient, magnitude,
direction, and a one-pixel sliding-window demonstration.

The 5×7 image is 20 on the left and 200 on the right. The selected patch has
three identical `[20, 20, 200]` rows: Gx=720, Gy=0, M=720, theta=0 degrees.
The Chapter 3 masks are applied directly as **correlation**, without flipping.
Image y increases downward; display vectors use `(Gx, -Gy)` with equal scaling.
Because Gy is exactly zero, the component triangle collapses to its horizontal
side and the zero-angle rays coincide. No artificial vertical component or
nonzero angle is drawn. Only interior pixels are demonstrated; border padding
and full output-map computation are outside this focused example.

Orange identifies Gx, purple Gy, cyan the active patch/boundary, yellow pairing
and direction, and green positive contributions/magnitude. Natural-language
captions use sequential whole-object fades with explicit removal of old text.
Typography uses Pango `Text`; LaTeX is not required.

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_sobel.py --preflight
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o SobelConvolution_review.mp4 sobel_convolution.py SobelConvolution
..\..\.venv\Scripts\python -B verify_sobel.py --video
```

Review: `media/videos/sobel_convolution/480p15/SobelConvolution_review.mp4`.
Report: `media/sobel/validation.json`, including the image, patch, masks, all
18 multiplications, responses, magnitude, direction, conventions and visual
scales. `--preflight` exercises every transition with frame-bound checks without
writing video. `--video` decodes every frame and checks 480p/15fps and 24–32 s.
`--qa` additionally creates a temporary contact sheet. Sobel text caches and
logs stay in `media/sobel/`; approved earlier scenes are not imported or edited.

## Hough circles: center loci, radius, gradient

`hough_circle.py` contains the final planned core scene, `HoughCircleTransform`.
A compass-like candidate circle stays incident on the first edge point while
its moving center traces a fixed-radius locus in the second coordinate plane.
Two more loci identify the common center; three additional edge points reinforce
the same intersection. The original image geometry remains visible while radius
slices transform into an affine 2.5D cone, introducing `A[a,b,r]`. An inward
image gradient then restricts a full locus to one candidate center per radius.

- True center `(1.5, 0.5)`, radius `2`; exact points from angles
  `180, 60, 300, 0, 120, 240` degrees. Both planes use Cartesian y-up coordinates.
- Each center locus is computed as `point + r*(cos(phi), sin(phi))`. No manually
  positioned curves or unrelated images are used. Radius layers are `1, 2, 3`.
- The synthetic intensity `0.5 + 0.5*tanh((2-distance)/0.15)` is bright inside.
  Its analytic gradient points inward, giving `center = p + r*unit_gradient`.
  The verifier checks this gradient independently with finite differences.
- Yellow/orange/purple connect the first three points to their loci; blue marks
  additional edge points; green identifies the common center and restricted vote.
  Captions use sequential whole-text fades, with explicit old-text removal.
- This is continuous geometric intuition, **not** a discretized/noisy circle
  detector. The 24 candidate dots illustrate alternatives for one point, not
  accumulator counts. Unknown polarity requires testing both gradient signs.
  No OpenCV tuning parameters or subsequent scene are included.

From `animations/chapter3/`:

```powershell
..\..\.venv\Scripts\python -B verify_hough_circle.py --preflight
..\..\.venv\Scripts\python -B -m manim -ql --disable_caching -o HoughCircleTransform_review.mp4 hough_circle.py HoughCircleTransform
..\..\.venv\Scripts\python -B verify_hough_circle.py --video
```

Review: `media/videos/hough_circle/480p15/HoughCircleTransform_review.mp4`.
Report: `media/hough_circle/validation.json`. The verifier records every edge
point, locus residual, independently solved common center, gradient, predicted
center, error and visual scale. `--preflight` checks frame bounds, updater cleanup
and 16:9 (allowing standard 854×480 raster rounding); `--video` decodes all frames
and checks 480p/15fps and 30–38 seconds. `--qa` creates temporary review images.
All caches, logs and QA outputs are scoped to this scene. Prior scenes are not
imported or edited.
