# Final integration audit — CV Interactive Demos

Verified on Windows, 2026-10-11. This report describes the actual local release checks; it does not claim that every possible UI state or production detector behavior was tested.

## Inventory and recovery

- Starting branch/HEAD: `feature/chapter3-manim-animations` at `540c6173b683f0f094642cdc1c3bbf7329f491ec`. The complete six-module `chapter3_webapp/` was untracked before this audit. Chapter 1 and seven approved Chapter 3 Manim scenes were present. No animation scene or numerical engine was rewritten.
- Initial project environments: root `.venv` (20,634 files, 725,196,897 file bytes) and `chapter1_webapp/.venv` (31,186 files, 914,183,981 file bytes), both Python 3.11.0. Original package inventories are saved as `root-venv-pip-freeze.txt` and `chapter1-venv-pip-freeze.txt` in the backup.
- Recovery backup: `C:\Users\ad\Downloads\codepython\project\CV-Interactive-Demos_backup_20261011_115333`. It contains 235 files / 42,466,820 file bytes: Chapter 1/3 source and docs, sample assets, original screenshot ZIPs, approved local animation media, and both package manifests; it excludes `.git`, venvs and caches. Key hashes were validated at creation; independently rechecked unchanged `chapter3_webapp/utils/ransac.py` and `tests/test_ransac.py` against the backup. To restore, copy the specific desired file from this backup to the corresponding repository path after checking current changes; do **not** blindly overwrite the whole worktree. The backup was not deleted.

## One canonical Python environment

- Final project-managed environment: repository-root `.venv`, Python 3.11.0. After consolidation: 21,277 files / 840,577,680 file bytes. `requirements.txt` pins direct dependencies: Streamlit 1.50.0, NumPy 2.2.6, OpenCV headless 4.12.0.88, Pillow 11.3.0, Plotly 6.3.0, Manim 0.19.0. The existing chapter manifests remain compatible; the root manifest and root interpreter are the documented installation/launch path.
- Root interpreter imported all six direct libraries and `python -m pip check` reported **No broken requirements found**. A separate temporary Python 3.11 venv installed the root manifest from scratch, imported all six libraries and passed `pip check`; it was subsequently removed. This validates a fresh install on this Windows host, not every platform.
- The old chapter-level `.venv` was removed **only after** root Chapter 1/3 tests, Manim verifiers, actual browser runs and independent clean installation passed. No running `python.exe` from that old path was found before removal. There is now one project-managed `.venv` under the repository. Do not create another chapter-local venv.
- Manim/PyAV and numerical preflight worked. This host emitted a pydub warning because an external `ffmpeg`/`avconv` executable was not detected. Some full render/export workflows may still require FFmpeg CLI; final 1080p videos were not regenerated.

## Tests and integration

| Check | Verified result |
| --- | --- |
| Chapter 1 baseline (old venv), then root before UI patch | 14/14 passing each time |
| Chapter 1 root after sidebar CSS fix | **14/14 passing** |
| Chapter 3 baseline and root after UI patches | **175/175 passing** each time; no skipped tests |
| RANSAC TLS and LSD geometry targeted regressions | Passed; included in the 175-test run |
| Root `pip check` and independent clean-install `pip check` | Both clean |
| Manim verifiers | `verify_review.py`, `verify_hough.py`, `verify_accumulator.py`, `verify_ransac.py`, `verify_lsd.py`, `verify_sobel.py --preflight`, `verify_hough_circle.py --preflight`: all exit 0 |
| Real Streamlit apps | Both launched with root interpreter; health endpoints previously returned HTTP 200 / `ok`; Edge sessions loaded both apps without application exceptions |

Run unit suites from their chapter directories with `..\.venv\Scripts\python.exe -B -m unittest discover -s tests`. Bare-mode Streamlit `missing ScriptRunContext` warnings during unit tests are expected. Neither app relies on the removed chapter venv.

## Browser matrix and targeted interactions

| App | Desktop 1920×1080 | Laptop 1366×768 | Tablet 768×1024 | Mobile 390×844 |
| --- | --- | --- | --- | --- |
| Chapter 1 | 8 module states checked | 8 checked | 8 checked | 8 checked |
| Chapter 3 | 6 Guided + 6 initial Lab views checked | 6 Guided checked | 6 Guided checked | 6 Guided checked |

The Chapter 3 four-viewport sweep was repeated after the UI patches: 30 screenshots, zero horizontal overflow, zero Streamlit exceptions, zero browser page/console errors. First Next/Previous was exercised in each module; each desktop Lab opened in the same session. Chapter 1's broad sweep covered all eight modules/four sizes. Its early screenshots exposed a sidebar overlap that simple `scrollWidth` measurement missed; after the global CSS fix, Pixel Explorer was visually rechecked at desktop/mobile and its sidebar's closed right edge measured exactly `0` (previously `59`/`29` px).

Focused Edge checks additionally verified:

- RANSAC Guided prediction gate at step 2, reveal gate at step 5, TLS step 8 (`0.1169 → 0.0950`, delta `−0.0218` shown downward/green), Presentation Mode chart height 562 → 722 px, Reset, and Lab prediction gate, one real iteration and all five phases Sample → Fit → Measure → Consensus → Update Best. Lab metrics for that seed showed `1/25` iterations and candidate/best consensus `33/37`; Lab reset worked.
- LSD Guided seed-prediction/reveal gate, BFS completion, and step 9 support rectangle. Plotly axes contained every rectangle corner with `y.scaleanchor = x`; the screenshot was visually inspected and no edge was clipped.
- Hough Circle Guided gates at steps 4 and 11, final detection at step 13, Presentation Mode, detected bin-center readout `(1.400, 0.600), r=2.000`, and no browser exception/overflow. The final view no longer shows a competing yellow candidate circle.
- Chapter 1 Gamma default grayscale and RGB-per-channel caption/three Plotly charts. Fourier reveal changed false → true, then reset to false after choosing a different sample image. Both had no browser errors.

Representative screenshots, visually opened during review: [Pixel Explorer](chapter1_webapp/docs/screenshots/final/pixel-explorer.png), [mobile Pixel Explorer](chapter1_webapp/docs/screenshots/final/pixel-explorer-mobile.png), [RANSAC Guided](chapter3_webapp/docs/screenshots/final/ransac-guided.png), [RANSAC TLS](chapter3_webapp/docs/screenshots/final/ransac-tls.png), [Hough](chapter3_webapp/docs/screenshots/final/hough-guided.png), [Sobel](chapter3_webapp/docs/screenshots/final/sobel-guided.png), [Canny](chapter3_webapp/docs/screenshots/final/canny-guided.png), [Circle final](chapter3_webapp/docs/screenshots/final/circle-final.png), [LSD rectangle](chapter3_webapp/docs/screenshots/final/lsd-support-rectangle.png). Existing Chapter 3 review PNGs referenced by module reports were preserved.

## Targeted UI fixes and educational accuracy

- RANSAC: TLS RMS delta now uses `TLS − candidate` with inverse semantic coloring and a short explanation: lower perpendicular RMS on the **same winning consensus** is better. Mobile progress is compact; current candidate, best consensus and TLS refit remain distinct.
- Hough Line: mobile progress copy is consistent and the last-vote marker is hidden when peak detection is shown, so it no longer competes with the winning bin.
- Sobel/Canny: consistent mobile progress wording; existing numerical conventions and gate logic retained.
- Hough Circle: final detected view de-emphasizes the previous candidate; the image-only panel fits small screens; accumulator colorbar is more compact.
- LSD: support-rectangle axis bounds expand to include real fitted corners with equal x/y scale, and the desktop teaching grid is taller. Regression tests assert containment and scale anchoring.
- Chapter 1: closed sidebar now respects Streamlit's 256 px translation, eliminating a 29–59 px overlay that clipped mobile text. Expanded sidebar width remains unchanged.
- No extra CV algorithm or frontend framework was added. RANSAC finite trials do not guarantee an optimum; Hough votes/bin centers are discretized; Sobel keeps signed gradients; Canny NMS uses four quantized directions; LSD NFA remains a teaching model rather than a calibrated reference implementation.

## Cleanup and documentation

- Removed six redundant Chapter 3 screenshot ZIPs after SHA-256 comparison of all 115 entries with extracted PNGs: **13,342,437 file bytes**. The originals are recoverable from the backup.
- Removed old `chapter1_webapp/.venv`: **914,183,981 file bytes**. Removed task-owned Playwright scripts, 50 broad-sweep screenshots, focused audit screenshots, `node_modules` and the temporary clean-install venv under `.final-audit-tools/`: **855,980,532 file bytes**. Final selected screenshots were copied and hash-checked first. The superseded `WORK_PROGRESS_CHECKPOINT.md` was also removed (10,866 file bytes) after this report was written. Total counted deleted file bytes: **1,783,517,816**. This is not the same as measured free-space change because installation/cache and unrelated disk activity also occur.
- C: free bytes: **182,265,978,880** at task start; **187,388,309,504** after these cleanups. Kept root `.venv`, recovery backup, original Chapter 3 review PNGs, tests/source/samples, and approved local Manim media. Only task-created Streamlit server sessions were stopped; task-owned Edge contexts were closed.
- Updated root, Chapter 1, Chapter 3 and Manim READMEs to explain unified setup and the actual six-module project. Root `.gitignore` now names `.final-audit-tools/` in addition to existing venv/cache/secrets/media exclusions. Nine final screenshots are documented; no temporary browser tool is intended for Git.

## Release and final repository state

- Intended release branch: `feature/cv-interactive-learning-lab`, starting from `540c6173b683f0f094642cdc1c3bbf7329f491ec` on `feature/chapter3-manim-animations`. Before branch creation, `origin` was confirmed as `https://github.com/Khanggolder/CV-Interactive-Demos.git`; the target branch did not exist locally or remotely, and the source remote branch matched local HEAD.
- Staging review: **182 files**, including 173 under `chapter3_webapp/` and 132 PNGs (existing report-linked review images plus nine selected final screenshots). The staged set had zero venv, `node_modules`, secret, ZIP, log or video paths; no project file exceeded 5 MB, `git diff --cached --check` was clean, and a targeted staged-content credential scan found no matches. Release commit SHA and remote push verification were **pending at the time this initial report was written**. Update this section with actual evidence after publishing; do not infer a push from a local commit. A commit cannot truthfully contain its *own* SHA, so the final report may identify the preceding implementation commit and be published in a follow-up documentation commit.
- No merge into `main`, force push, reset, rebase or remote branch deletion is authorized or performed.

## Remaining limitations

- Browser sweeps covered every module and four viewport sizes, but did not mechanically exhaust every Guided step, Lab parameter combination or Presentation state in all six modules. Deep flows were exercised for RANSAC, LSD and Hough Circle as listed above; unit tests cover the numerical engines. This is a teaching app audit, not an exhaustive formal UI proof.
- Only the selected Chapter 1 sidebar state was visually rechecked after the global CSS fix; the same selector applies to all Chapter 1 modules. Broad pre-fix screenshots are not used as final review evidence.
- The standalone FFmpeg CLI was not detected; check it before a workflow requiring high-quality Manim video export. No full final videos were rerendered.
