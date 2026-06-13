# Experiment Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-file XeLaTeX report project under `report/` for the semester robot experiments.

**Architecture:** `report/main.tex` owns document setup, cover, table of contents, bibliography, and chapter inclusion. Each experiment lives in its own `report/chapters/*.tex` file; shared figures and code excerpts live under `report/figures/` and `report/code/`. The report is written from local evidence in `daran/`, with safe simulation or static-output verification only.

**Tech Stack:** LaTeX/XeLaTeX, `ctexart`, `graphicx`, `amsmath`, `booktabs`, `listings`, local Python scripts/notebooks/images from `daran/`.

---

### Task 1: Create Report Project Skeleton

**Files:**
- Create: `report/main.tex`
- Create: `report/chapters/00_abstract.tex`
- Create: `report/chapters/01_environment.tex`
- Create: `report/chapters/02_cartesian_line_motion.tex`
- Create: `report/chapters/03_pick_and_place.tex`
- Create: `report/chapters/04_camera_calibration.tex`
- Create: `report/chapters/05_color_visual_grasp.tex`
- Create: `report/chapters/06_yolo_visual_grasp.tex`
- Create: `report/chapters/07_conclusion.tex`
- Create: `report/figures/.gitkeep`
- Create: `report/code/.gitkeep`
- Create: `report/references.bib`
- Create: `report/README.md`

- [ ] **Step 1: Create directories**

Run: `mkdir -p report/chapters report/figures report/code`

Expected: directories exist.

- [ ] **Step 2: Write `report/main.tex`**

Use `ctexart`, configure A4 layout, Chinese fonts through XeLaTeX-compatible defaults, figure/table/code packages, and include all chapter files with `\input{chapters/...}`.

- [ ] **Step 3: Write empty chapter files with section anchors**

Each chapter file should compile independently through `main.tex` and contain a clear top-level `\section{...}`.

- [ ] **Step 4: Write compile instructions**

Create `report/README.md` with:

```markdown
# 机器人综合实验报告

编译方式：

```bash
cd report
latexmk -xelatex main.tex
```

如果没有 `latexmk`，可使用：

```bash
cd report
xelatex main.tex
xelatex main.tex
```
```

- [ ] **Step 5: Verify skeleton compiles or record missing TeX toolchain**

Run: `cd report && latexmk -xelatex -interaction=nonstopmode main.tex`

Expected: PDF generated as `report/main.pdf`, or a clear message that the TeX toolchain is unavailable.

### Task 2: Gather Evidence From Repository

**Files:**
- Read: `CLAUDE.md`
- Read: `environment.yml`
- Read: `daran/report.md`
- Read: `daran/task1/*`
- Read: `daran/task2/pick_and_place.py`
- Read: `daran/task3/camera_calibration.py`
- Read: `daran/task3/pose_projection.py`
- Read: `daran/task3/visual_grasp.py`
- Read: `daran/color_pick_place/*.py`
- Read: `daran/color_pick_place/yolo/*.py`
- Modify: chapter files under `report/chapters/`
- Copy or reference selected images into `report/figures/`

- [ ] **Step 1: Inspect environment and course notes**

Run: `sed -n '1,220p' CLAUDE.md` and `sed -n '1,220p' environment.yml`.

Expected: extract environment name, Python version, core packages, and hardware warnings.

- [ ] **Step 2: Inspect task 1 materials**

Run: `find daran/task1 -maxdepth 1 -type f -print`.

Expected: identify line-motion notebook/script outputs and existing result images or GIFs.

- [ ] **Step 3: Inspect task 2 materials**

Run: `sed -n '1,240p' daran/task2/pick_and_place.py`.

Expected: extract pick-place pose sequence, motion API use, and any simulation/result asset names.

- [ ] **Step 4: Inspect task 3 materials**

Run: `sed -n '1,260p' daran/task3/camera_calibration.py`, `sed -n '1,260p' daran/task3/pose_projection.py`, and `sed -n '1,260p' daran/task3/visual_grasp.py`.

Expected: extract camera calibration flow, coordinate projection method, and visual grasp pipeline.

- [ ] **Step 5: Inspect color visual grasp materials**

Run: `find daran/color_pick_place -maxdepth 1 -type f -print` and read the main Python files.

Expected: extract intrinsics/extrinsics files, color detector flow, teach calibration, and main grasp sequence.

- [ ] **Step 6: Inspect YOLO materials**

Run: `find daran/color_pick_place/yolo -maxdepth 2 -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.txt' -o -name '*.json' \) -print`.

Expected: extract dataset structure, class names, training config, validation/detection scripts, and model files.

### Task 3: Draft Chapters

**Files:**
- Modify: `report/chapters/00_abstract.tex`
- Modify: `report/chapters/01_environment.tex`
- Modify: `report/chapters/02_cartesian_line_motion.tex`
- Modify: `report/chapters/03_pick_and_place.tex`
- Modify: `report/chapters/04_camera_calibration.tex`
- Modify: `report/chapters/05_color_visual_grasp.tex`
- Modify: `report/chapters/06_yolo_visual_grasp.tex`
- Modify: `report/chapters/07_conclusion.tex`
- Modify: `report/references.bib`

- [ ] **Step 1: Draft abstract and overview**

Write the report objective, six-experiment progression, and final integrated capability.

- [ ] **Step 2: Draft environment chapter**

Cover conda environment, key libraries, robot hardware connection, and repo structure.

- [ ] **Step 3: Draft Cartesian line motion chapter**

Cover DH model, FK/IK, Cartesian interpolation, joint limit protection, and task 1 result assets.

- [ ] **Step 4: Draft pick-and-place chapter**

Cover pre-grasp, descend, close gripper, lift, transfer, place, and return phases.

- [ ] **Step 5: Draft camera calibration chapter**

Cover pinhole model, distortion, chessboard image collection, calibration outputs, undistortion, and workspace transform.

- [ ] **Step 6: Draft color visual grasp chapter**

Cover color threshold segmentation, contour center extraction, pixel-to-world transform, and grasp execution.

- [ ] **Step 7: Draft YOLO visual grasp chapter**

Cover dataset, classes, train/val split, model training or loading, detector output, and comparison to color segmentation.

- [ ] **Step 8: Draft conclusion**

Summarize system evolution, error sources, limitations, and future improvements.

### Task 4: Add Figures, Tables, and Code Excerpts

**Files:**
- Create/modify: files under `report/figures/`
- Create/modify: files under `report/code/`
- Modify: chapter files under `report/chapters/`

- [ ] **Step 1: Select static figures**

Copy representative existing images from `daran/task1`, `daran/task2`, `daran/task3/calibration_output`, and `daran/color_pick_place` into `report/figures/` using descriptive names.

- [ ] **Step 2: Add DH and calibration tables**

Add LaTeX tables for DH parameters, camera calibration outputs, and chapter-level experiment assets where useful.

- [ ] **Step 3: Add short code excerpts**

Add only compact excerpts that explain key implementation points. Avoid dumping full source files.

- [ ] **Step 4: Verify all figure references exist**

Run: `rg -n "\\includegraphics" report/chapters report/main.tex`.

Expected: every referenced file exists under `report/figures/`.

### Task 5: Compile and Polish

**Files:**
- Modify: `report/main.tex`
- Modify: `report/chapters/*.tex`
- Modify: `report/README.md`

- [ ] **Step 1: Compile with XeLaTeX**

Run: `cd report && latexmk -xelatex -interaction=nonstopmode main.tex`.

Expected: `report/main.pdf` is generated without fatal errors.

- [ ] **Step 2: Fix compile errors**

If compilation fails, inspect `report/main.log`, fix missing packages, invalid LaTeX syntax, or bad figure paths, then rerun the compile command.

- [ ] **Step 3: Check cross references**

Run the compile command a second time.

Expected: no unresolved reference warnings for locally defined figures/tables.

- [ ] **Step 4: Check git scope**

Run: `git status --short report docs/superpowers/plans/2026-06-13-experiment-report.md`.

Expected: only the new report project and the implementation plan are changed by this work.

- [ ] **Step 5: Commit report work**

Run:

```bash
git add report docs/superpowers/plans/2026-06-13-experiment-report.md
git commit -m "docs: draft robot experiment report"
```

Expected: commit contains the report project and the plan, without unrelated existing workspace changes.
