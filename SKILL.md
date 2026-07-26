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
