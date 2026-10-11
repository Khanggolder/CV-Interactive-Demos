# RANSAC MVP — browser UI/UX audit

Audit date: 2026-10-10. Scope: `chapter3_webapp/` only. No commit, push, new Chapter 3 algorithm, or global dependency change.

## Method and visual evidence

Launched the real Streamlit app with the existing Chapter 1 Python environment at `http://localhost:8514/`. `/_stcore/health` returned `ok`, and Edge loaded the actual app page. Used temporary `playwright-core` with the installed Microsoft Edge; no browser binary was downloaded. Tested 1920×1080, 1366×768, 768×1024, and 390×844. Inspected the screenshots visually as well as DOM overflow, clipping, Streamlit exceptions, and browser console/page errors. Final browser runs had no horizontal overflow, clipping flags, console/page errors, or Streamlit exception elements. The local server and browser sessions were closed after testing.

Representative screenshots (8 retained, 768,625 bytes):

- [Desktop Guided](docs/screenshots/desktop-initial.png), [tablet Guided](docs/screenshots/tablet-initial.png), [mobile Guided](docs/screenshots/mobile-initial.png)
- [Perpendicular distance and ε band](docs/screenshots/guided-step-4.png), [TLS refit](docs/screenshots/guided-step-8.png), [OLS comparison](docs/screenshots/guided-step-9.png)
- [Many Outliers Lab result](docs/screenshots/many-outliers-all.png), [mobile Lab with sidebar closed](docs/screenshots/mobile-lab-confirmed.png)

## Verified findings and smallest fixes

| Finding | Change | Verification |
| --- | --- | --- |
| Header obscured the Chapter 3 kicker; mobile sidebar initially covered the page; tablet navigation/preset labels wrapped. | Auto-collapse sidebar, increase top padding, make Guided navigation three equal columns, widen the preset button, retain sidebar brand subtitle contrast. | Re-inspected four viewports and both modes; no horizontal overflow or unreadable labels. |
| Guided could advance without a prediction or reveal; step 5's question was only a prompt. | Disable Next at steps 2 and 5 until the required actions; add a short membership prediction before enabling Reveal; use callbacks so Streamlit reruns update controls immediately; Reset clears both predictions and reveal. | Browser completed all 10 steps, Previous/Next/Reset, and checked both disabled gates; AppTest regression passed. |
| Loading a Lab preset changed controls but left the old point cloud active. | Generate the matching active dataset in the preset callback. | AppTest and browser: each preset loaded the stated point count/configuration and reset the experiment. |
| At ε=0, floating-point roundoff could exclude the two points defining a candidate line. | Use `1e-12` only as a distance-comparison tolerance; validate ε in direct pair evaluation. | Minimum browser case: 2 points, ε=0, score 2/2; unit regression passed. |
| A generated point at maximum noise could fall outside the chart's fixed y range. | Expand chart bounds to include every actual point with a small margin. | Extreme-parameter chart and visualization regression passed. |
| Rapid reruns briefly triggered Plotly's optional pandas validator while pandas was partially initialized. | Import pandas before Plotly's graph-object validators in the visualization module. | Subsequent browser runs and 17 tests had no runtime exception. |

## Interaction and numerical results

- Guided: steps 1–10, active stepper, Previous, Reset, prediction/reveal gates, ε band, perpendicular-distance plot, score, TLS refit, and OLS comparison passed. The approved seeded fixture produced sample pairs `(14, 6)`, `(5, 2)`, `(2, 11)` and consensus scores `3`, `7`, `14`; best trial 3. Refit uses that best consensus, not generated labels.
- Lab: Clean Data `33/37` (25 iterations), Many Outliers `24/60` (80), Wrong Epsilon too small `11/42` (45), too large `36/42` (45). For each, tested one iteration, batch of 10, run all, membership inspection, and reset. Revealed the preset-comparison table. Scores and plots responded to the active dataset rather than stale preset controls.
- Extremes: 2 inliers/0 outliers, noise 0, ε 0, 1 iteration gave `2/2`; 100 inliers/100 outliers, noise 0.5, ε 1.2, 200 iterations gave best consensus `145` (first trial `127`). Changing ε to 0 after running recomputed the best consensus to `2`; changing seed and regenerating reset the experiment. Tablet/mobile Lab navigation was checked with the sidebar closed.
- Regression: `python -m unittest discover -s tests -v` → **17/17 passed**, including Streamlit AppTest, geometry, deterministic RANSAC, and visualization tests. The bare-mode AppTest `missing ScriptRunContext` warning was expected. Final health check: `ok`. `git diff --check` produced no output; because the Chapter 3 tree is untracked, a separate trailing-whitespace scan of Chapter 3 Python/Markdown files also produced no matches.

## Remaining limitations

This MVP demonstrates a single 2D line using a fixed seeded Guided example; it does not let students drag points or fit multiple structures. Lab predictions are coarse categories, not a calibrated probability estimate. Long mobile pages require vertical scrolling, but controls remain visible and usable. RANSAC's finite-iteration result still depends on seed, ε, noise, and outlier proportion; the app states this rather than implying a guarantee.

## Safe cleanup

Before cleanup, the temporary `chapter3_webapp/.ui-audit-tools/` folder contained 15,942,987 file bytes (including temporary `playwright-core` and 22 screenshots), and the newly generated `chapter3_webapp/tests/__pycache__/test_visualization.cpython-311.pyc` contained 2,113 bytes. Both exact paths were verified inside the workspace and removed. **Total file bytes removed: 15,945,100.** C: available space immediately before/after cleanup was 188,559,081,472 / 188,575,387,648 bytes (allocation-based gain: 16,306,176 bytes). Other pre-existing caches were preserved. The audit's Streamlit PID 4400 was confirmed to own port 8514 and stopped; no matching Playwright Edge/Node processes remained. The eight final screenshots above were retained. No global software, shared browser cache/profile, tracked file, existing Manim scene, or review video was removed.

At completion, `git status --short` reports `?? chapter3_webapp/` on branch `feature/chapter3-manim-animations`; no files were staged, committed, or pushed.
