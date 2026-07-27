# README Design

## Goal

Create a bilingual GitHub README for EEG One-Stop Skill that helps mouse
EEG/EMG researchers understand what the skill does, install it, invoke it, and
interpret its outputs without overstating the scientific evidence.

## Audience

- Researchers processing mouse EEG/EMG recordings with Codex.
- Contributors reviewing or extending the deterministic Python components.

## Language And Order

Provide the complete Chinese README first, followed by a complete English
version. Keep headings parallel so readers can move between the two versions.

## Content Structure

1. Project title and concise description.
2. Core capabilities and intended use cases.
3. Safety and reproducibility principles.
4. Requirements and installation.
5. Example Codex invocation and guided workflow.
6. Expected outputs and evidence trail.
7. Repository structure and development tests.
8. Current limitations and scientific disclaimer.
9. License status, stated accurately from repository evidence.

## Accuracy Rules

- Describe only capabilities present in `SKILL.md`, Python modules, or tests.
- Do not advertise `scripts/run_pipeline.py` because it is not currently in
  the repository.
- Do not add CI, coverage, release, license, or compatibility badges that are
  not backed by current repository metadata.
- Do not claim automatic dependency installation.
- State that source EDF/FIF data remain unchanged and that review gates require
  user confirmation.
- Identify Hilbert sleep-stage outputs as candidate labels requiring expert
  validation.
- State that single-animal EEG evidence cannot establish antidepressant
  efficacy.

## Installation And Usage

Show a Git clone command and a Codex skill-directory installation example.
Document the Python range and dependencies from `pyproject.toml` and
`environment.yml`. Use a conversational Codex prompt as the primary usage
example because the repository does not yet expose a complete command-line
pipeline entry point.

## Verification

Before publication:

- Run the complete test suite with the repository's known Python environment.
- Validate the skill package with the available skill validator.
- Check Markdown headings, internal file references, and command paths.
- Confirm the Git worktree contains only the intended documentation changes.
- Push to `origin/main`, then verify local and remote commit identifiers match.

## Out Of Scope

- Changing signal-processing code, tests, dependencies, or skill behavior.
- Adding screenshots, generated figures, benchmark claims, or a license file.
- Claiming biological efficacy or validated sleep-stage ground truth.
