---
name: eeg-one-stop-skill
description: Guide and run a reproducible mouse EEG workflow from EDF/FIF import through direct concatenation, user-supplied Markers, bipolar reference, artifact review, line-noise filtering, paper-aligned Bandpower, Hilbert EEG+EMG candidate sleep staging, and evidence-backed chart reporting. Use this skill whenever a user asks to preprocess mouse EEG/EMG, analyze N2O EEG, build Bandpower or time-frequency outputs, review 50/100 Hz interference, or produce Hilbert Wake/NREM/REM candidate results, even when they do not explicitly request a pipeline.
compatibility: Windows 10/11; user-managed Python 3.11-3.13 environment with MNE, NumPy, SciPy, pandas, matplotlib, and JupyterLab.
---

# EEG One-Stop Skill

Use a conversational layer for user decisions and the bundled Python modules
for deterministic processing. Never replace missing experimental information
with a guess.

## Start Here

1. Ask the user to select the result export folder. This must be the first
   question and must happen before opening any EDF or FIF file.
2. Run the detect-only environment check. If anything is missing, read
   `references/environment-setup.md`, provide manual installation guidance,
   and stop until the user reports the environment name and Python path.
3. Complete the import checklist. Require file order, channel identities,
   bipolar formulas, and every Marker name, value, timing basis, source file,
   and meaning.
4. Use `scripts/run_pipeline.py` for deterministic work once that command is
   available. Do not rewrite signal-processing logic inside the conversation.

## Non-Negotiable Boundaries

- Do not run package managers or installers for Anaconda, Python, MNE,
  JupyterLab, Git, or other user-side software.
- Concatenate files only in the order the user confirms. Do not detect or trim
  overlap/gaps in this version.
- Preserve an immutable checkpoint before each destructive-looking decision.
  BAD annotations exclude analysis windows but never delete samples.
- Pause for user confirmation at Marker, artifact, 100 Hz, and Hilbert gates.
- Keep `Artifact`, `Uncertain`, and `Boundary_Unscored` labels unchanged.
- Treat Hilbert sleep states as candidate labels requiring expert validation.

## Reporting Contract

Before producing the final report, read
`references/chart-analysis-contract.md`. Inspect every generated chart with
Codex image analysis and cross-check it against its declared CSV/JSON source.
Explain every chart beside the image and keep chart names limited to the
content shown. Put experimental and manual-review warnings in conclusions,
not in chart titles.

The overall conclusion must separately answer whether the run shows an
electrophysiological change, whether its direction is consistent with the
cited paper, and whether it supports an obvious antidepressant effect. A
single-animal EEG-only run must label the antidepressant claim as
`证据不足`.


## Artifact Candidate Gate

Run this gate only after the bipolar checkpoint and pre-artifact byte-identical
copy have been frozen.

1. Generate `artifact_method_explanation()` and show its full text to the user
   before calling `detect_artifact_candidates()`. The explanation must cover
   the approved 5-second/MAD/6-threshold method and teach how common artifacts
   look: high-amplitude transients, movement or cable disturbance, muscle
   contamination, flatline or dropout, clipping or saturation, and periodic
   50 Hz or 100 Hz line noise.
2. State that candidate detection does not delete samples or prove an artifact.
   Do not proceed until the user confirms they have read the method.
3. Detect candidates, attach only `CANDIDATE_` annotations, and generate every
   candidate image plus the CSV/JSON review bundle. Never silently cap or sample
   the image set.
4. Use `view_image` on every one of the generated candidate images. Cross-check
   each image with its CSV/JSON metrics. For every image, record a conclusion
   (`支持伪迹`, `不支持伪迹`, or `无法判断`), the possible artifact type, the
   judgment reason, and the exact image evidence. Do not infer the conclusion
   from the filename or metric alone. Do not label high EMG alone as artifact;
   it may be genuine wake movement.
5. Call `write_completed_artifact_reviews()` only after every image has all four
   review fields. Show the completed report and tell the user how many candidates
   are supported, unsupported, or unresolved.
6. Ask one batch question: whether to exclude the Codex-supported candidates.
   Use `exclude_supported` only after an explicit yes; otherwise use `retain_all`.
   Candidates marked `无法判断` or `不支持伪迹` must never become `BAD_auto_`.
7. Freeze the decision output under `数据冻结/05_仅伪迹标注后` without
   overwriting. Preserve the pre-artifact FIF, candidate images, algorithm
   reasons, completed per-image review, user decision, Annotations, and SHA256.

The 50 Hz and 100 Hz decision belongs to the later line-noise gate. Do not treat
a periodic line-noise peak as a segment to delete here.


用户可见的伪迹审核说明必须明确包含“常见伪迹”的判断方法，并对每一张图片
分别给出伪迹类型、判断理由和图像证据。结论不充分时必须标记“无法判断”；
不得仅凭高EMG把窗口认定为伪迹。50 Hz和100 Hz周期性干扰交由后续工频审核。
