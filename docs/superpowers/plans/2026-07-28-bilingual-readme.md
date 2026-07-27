# Bilingual README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an accurate Chinese-first bilingual README for EEG One-Stop Skill.

**Architecture:** Add one root `README.md` whose Chinese and English halves use parallel headings. Derive every capability, command, dependency, and limitation from the tracked skill, modules, tests, and package metadata, then verify the documentation and repository before pushing `main`.

**Tech Stack:** Markdown, Git, Python 3.11-3.13, pytest, Codex skill validator

## Global Constraints

- Do not modify signal-processing code, tests, dependencies, or skill behavior.
- Do not mention `scripts/run_pipeline.py` as an available entry point.
- Do not add unsupported CI, coverage, release, license, or compatibility badges.
- State that dependencies are installed manually and source EDF/FIF files remain unchanged.
- Present Hilbert sleep-stage outputs only as candidate labels requiring expert review.
- Do not claim that single-animal EEG establishes antidepressant efficacy.

---

### Task 1: Write The Bilingual README

**Files:**
- Create: `README.md`
- Reference: `SKILL.md`
- Reference: `pyproject.toml`
- Reference: `environment.yml`
- Reference: `agents/openai.yaml`

**Interfaces:**
- Consumes: tracked repository capabilities, requirements, and safety boundaries.
- Produces: a GitHub-renderable, Chinese-first bilingual project guide.

- [ ] **Step 1: Create the Chinese README half**

Write these sections in order: project description, core capabilities, design principles, requirements, installation, Codex invocation, workflow, outputs, repository structure, development verification, limitations, and license status. Use the repository clone URL and `%USERPROFILE%\.codex\skills\eeg-one-step-skill` as the Windows installation example.

- [ ] **Step 2: Create the parallel English README half**

Translate the complete Chinese content faithfully. Preserve command blocks and scientific limitations, and keep the heading order parallel.

- [ ] **Step 3: Run Markdown and repository-reference checks**

Run:

```powershell
git diff --check
rg -n "run_pipeline|ground truth|Ground Truth|抗抑郁|antidepressant|pip install|conda install" README.md
```

Expected: `git diff --check` exits 0; `run_pipeline` is absent; installation commands are explicitly presented as manual user actions; scientific limitations are present in both languages.

- [ ] **Step 4: Review the rendered structure mechanically**

Run:

```powershell
rg -n "^#|^##|^###" README.md
```

Expected: one project H1 followed by complete Chinese sections, a language divider, and complete parallel English sections with no empty headings.

- [ ] **Step 5: Commit the README**

```powershell
git add -- README.md
git commit -m "docs: add bilingual project readme"
```

Expected: one commit containing only `README.md`.

### Task 2: Verify And Publish

**Files:**
- Verify: `README.md`
- Verify: complete tracked repository

**Interfaces:**
- Consumes: committed documentation and existing test suite.
- Produces: verified commits on `origin/main`.

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
D:\pythonmne\python.exe -m pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run the installed skill validator**

Locate `quick_validate.py` under the installed skill tooling, then run it against the repository.

Expected: exit code 0 and `Skill is valid!`.

- [ ] **Step 3: Check intended changes and history**

Run:

```powershell
git status --short --branch
git log -3 --oneline --decorate
git diff origin/main..HEAD --stat
```

Expected: clean worktree, branch ahead only by the design, plan, and README documentation commits.

- [ ] **Step 4: Push through the active Clash Verge proxy**

Run:

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin main
```

Expected: push succeeds without changing global Git configuration.

- [ ] **Step 5: Verify the remote commit**

Run:

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 ls-remote origin refs/heads/main
git rev-parse HEAD
```

Expected: the remote `refs/heads/main` SHA equals local `HEAD` and `git status --short --branch` reports no divergence.
