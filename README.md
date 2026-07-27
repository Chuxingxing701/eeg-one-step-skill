# EEG One-Stop Skill

面向 Codex 的可复现小鼠 EEG/EMG 分析 skill，从 EDF/FIF 导入一直引导到 Bandpower、Hilbert 候选睡眠分期和证据化结果报告。

[English](#english)

## 项目简介

EEG One-Stop Skill 将需要研究者判断的步骤保留为明确的审核门，并用 Python 模块完成可重复的计算。它适用于小鼠 EEG/EMG、N2O EEG、工频干扰审查、Bandpower 分析，以及探索性的 Hilbert Wake/NREM/REM 候选分期。

这不是一个点击后自动给出生物学结论的黑箱。文件顺序、通道身份、双极公式、Marker、实验阶段、伪迹排除和 100 Hz 处理等关键决定都需要用户提供或确认。

## 核心能力

- 检测 Python、MNE、NumPy、SciPy、pandas、Matplotlib 和 pytest 环境，不静默安装软件。
- 按用户确认的顺序直接导入并拼接 EDF/FIF，核对采样率、通道结构和测量时间。
- 只保留双极公式涉及且经用户确认的分析通道，并记录 EEG/EMG 类型和排除通道。
- 解析用户提供的 Marker，检查来源文件、时间基准、范围、冲突和实验日志顺序。
- 建立双极 EEG/EMG，冻结不可覆盖的 FIF、配置、Annotations、软件版本和 SHA256 证据。
- 用 5 秒窗口和稳健 MAD 阈值生成 `CANDIDATE_` 伪迹候选；候选不等于已确认伪迹，必须逐图审核。
- 分别计算 EEG 与 EMG 的滤波前后 PSD，审查 50 Hz 和 100 Hz；100 Hz 是否处理必须由用户明确决定。
- 计算论文参数对齐的 Bandpower：5 秒分析窗、2.5 秒步长、2 秒 Hann-Welch 子窗和 50% 重叠。
- 生成探索性的 Hilbert EEG+EMG 候选分期，并保留 `Artifact`、`Uncertain` 和 `Boundary_Unscored`。
- 逐图交叉核对 PNG 与 CSV/JSON 来源，形成带方法、参数、人工决定和局限性的结果报告。

## 设计原则

- **原始数据不变**：不覆盖或修改源 EDF/FIF；关键阶段保存不可变检查点。
- **决定可追溯**：Marker、伪迹、50/100 Hz 和 Hilbert 均有显式审核门。
- **候选不是事实**：`CANDIDATE_` 和 Hilbert 睡眠标签都需要有经验的人工复核。
- **逻辑排除而非删除**：`BAD_auto_` Annotation 只在后续分析中排除窗口，不删除采样点。
- **论文参数与工程选择分开**：论文报告的 PSD/Bandpower 参数与零相位 FIR 陷波等工程处理分别标注。
- **证据先于结论**：图表必须结合其 CSV/JSON 数据源解释，不能仅凭文件名或图形外观下结论。

## 环境要求

- Codex 或支持 Codex skill 的运行环境
- Python `>=3.11,<3.14`
- MNE `>=1.8`
- NumPy `>=2.0`
- SciPy `>=1.13`
- pandas `>=2.2`
- Matplotlib `>=3.9`
- pytest `>=8`（开发与验证）
- JupyterLab `>=4`（可选的交互环境）

## 安装

### 1. 安装 skill

在 PowerShell 中执行：

```powershell
git clone https://github.com/Chuxingxing701/eeg-one-step-skill.git "$env:USERPROFILE\.codex\skills\eeg-one-step-skill"
```

重启 Codex 或开始一个新任务，使其重新发现 skill。

### 2. 手动创建 Python 环境

本 skill 只检测环境，不会替你运行包管理器或安装器。请由你本人在仓库目录中执行：

```powershell
conda env create -f environment.yml
conda activate mouse-eeg
```

也可以按照 [`references/environment-setup.md`](references/environment-setup.md) 手动安装并把环境名称、Python 解释器路径和 Conda 安装位置告诉 Codex。

## 使用方式

在 Codex 中可以直接提出任务，例如：

```text
使用 $eeg-one-stop-skill，引导我完成这批小鼠 EEG/EMG 的分析。
从选择结果导出文件夹开始，在读取任何 EDF 前先核对文件顺序、通道、双极公式和 Marker。
```

也可以描述具体目标：

```text
检查这组 N2O 小鼠 EEG 的 50/100 Hz 干扰，完成 Bandpower，
然后在我确认 smoke test 后生成 Hilbert 候选睡眠分期和证据化报告。
```

skill 会先要求选择结果导出文件夹，然后检测环境。实验信息不足时会停止并询问，不会猜测文件顺序、通道含义、Marker 或实验阶段。

## 工作流程

1. **选择输出目录**：在打开 EDF/FIF 前确定结果保存位置。
2. **环境检测**：报告当前解释器和依赖；缺失时给出手动安装说明并停止。
3. **导入清单**：确认文件顺序、通道身份与类型、双极公式，以及每个 Marker 的来源和含义。
4. **导入与双极参考**：直接拼接用户确认的文件，选择分析通道，建立双极信号并冻结检查点。
5. **伪迹候选审核**：解释方法、生成全部候选图片，由 Codex 逐图结合指标审核，再由用户决定是否排除有支持证据的候选。
6. **工频审核**：比较滤波前 PSD；50 Hz 按已批准流程处理，100 Hz 必须展示证据并等待用户决定；随后复核滤波后 PSD。
7. **Bandpower**：按用户提供的实验阶段计算逐窗绝对和相对功率，保留 Artifact 与边界窗口。
8. **Hilbert 候选分期**：先说明探索性边界并运行用户指定的 smoke test；两次确认后才处理目标时段。
9. **结果报告**：逐图读取真实 PNG 并与声明的 CSV/JSON 交叉核对，最后汇总参数、决定、SHA256 和局限性。

## 输出与可追溯性

根据完成的阶段，输出可包括：

- 不可覆盖的 FIF 检查点及 SHA256
- 输入文件、完整通道清单、保留/排除通道和软件版本
- Marker 配置、Annotations 和用户决定记录
- 伪迹候选图片、指标 CSV/JSON 和逐图审核结果
- 滤波前后 PSD、局部峰值数据与 50/100 Hz 决定
- Bandpower 逐窗 CSV、参数 JSON、阶段比较与时频图
- Hilbert 候选逐窗 CSV、Bout CSV、阶段汇总和人工复核清单
- 图表来源清单、逐图解释与最终 Markdown 报告

具体输出取决于用户完成并确认了哪些审核门。未完成审核时，相关结果必须保持待处理状态。

## 仓库结构

```text
eeg-one-step-skill/
|-- SKILL.md                         # skill 工作流与审核门
|-- agents/openai.yaml               # Codex 展示与调用配置
|-- scripts/eeg_pipeline/            # 确定性的 EEG 处理模块
|-- references/
|   |-- environment-setup.md         # 手动环境安装指南
|   `-- chart-analysis-contract.md   # 逐图分析与结论契约
|-- assets/checklist_schema.json     # 输入清单结构
|-- tests/                            # 行为、安全边界和回归测试
|-- environment.yml                  # Conda 环境定义
`-- pyproject.toml                   # Python 包与 pytest 配置
```

当前仓库提供可复用的 Python 组件和 Codex 引导流程，尚未提供完整的一键命令行流水线入口。

## 开发与验证

激活依赖完整的环境后运行：

```powershell
python -m pytest -q
```

测试覆盖环境检测、输入清单、EDF/FIF 导入、分析通道选择、Marker、双极参考、伪迹审核、工频证据、Bandpower、Hilbert 候选分期、工作区状态和结果报告边界。

## 当前限制与科研声明

- 该项目面向研究工作流，不用于临床诊断、治疗决策或医疗器械用途。
- 文件只按用户确认的顺序直接拼接；当前版本不自动检测或裁剪重叠与间隙。
- 伪迹候选检测是排序与审核辅助，不是自动伪迹诊断。高 EMG 也可能是真实清醒活动。
- Hilbert 的 `Wake_candidate`、`NREM_candidate` 和 `REM_candidate` 是探索性候选标签，不是 Ground Truth，尤其需要人工复核 REM。
- Bandpower 独立于未经验证的睡眠候选标签，不能用候选分期重写或丢弃 Bandpower 窗口。
- 单只动物的 EEG 变化可以用于描述电生理趋势，但不能证明抗抑郁疗效；仍需要更多动物、重复实验、行为学分析和有经验的人工审核。

## 许可证

仓库当前未包含 `LICENSE` 文件。除非作者另行授权，否则请勿假定代码已按某个开源许可证授权；复制、修改或再分发前请联系仓库作者。

---

<a id="english"></a>

## English

EEG One-Stop Skill is a reproducible mouse EEG/EMG workflow for Codex. It guides a study from EDF/FIF import through Bandpower, Hilbert candidate sleep staging, and evidence-backed result reporting.

## Overview

The skill keeps decisions that require researcher judgment behind explicit review gates and uses Python modules for repeatable computation. It is intended for mouse EEG/EMG, N2O EEG, line-noise review, Bandpower analysis, and exploratory Hilbert Wake/NREM/REM candidate staging.

It is not a black box that produces a biological conclusion after one click. File order, channel identity, bipolar formulas, Markers, experimental phases, artifact exclusion, and the 100 Hz decision must be supplied or confirmed by the user.

## Core Capabilities

- Detect the Python, MNE, NumPy, SciPy, pandas, Matplotlib, and pytest environment without silently installing software.
- Import and directly concatenate EDF/FIF files in the user-confirmed order while checking sampling rates, channel structure, and measurement times.
- Retain only user-confirmed analysis channels used by the bipolar formulas, with recorded EEG/EMG types and excluded channels.
- Resolve user-supplied Markers and validate source files, timing bases, ranges, conflicts, and experimental-log order.
- Build bipolar EEG/EMG signals and freeze non-overwriting FIF, configuration, Annotation, software-version, and SHA256 evidence.
- Generate `CANDIDATE_` artifact windows using robust MAD thresholds over 5-second windows; candidates require per-image review and are not confirmed artifacts.
- Compute separate pre/post-filter EEG and EMG PSD evidence for 50 Hz and 100 Hz review; the user must explicitly decide whether to process 100 Hz.
- Compute paper-aligned Bandpower with 5-second analysis windows, a 2.5-second step, 2-second Hann-Welch subwindows, and 50% overlap.
- Generate exploratory Hilbert EEG+EMG candidate stages while retaining `Artifact`, `Uncertain`, and `Boundary_Unscored`.
- Cross-check each PNG against its CSV/JSON source and produce a report containing methods, parameters, human decisions, and limitations.

## Design Principles

- **Source data remain unchanged:** source EDF/FIF files are never overwritten or modified; immutable checkpoints preserve key stages.
- **Decisions are traceable:** Marker, artifact, 50/100 Hz, and Hilbert steps use explicit review gates.
- **Candidates are not facts:** `CANDIDATE_` windows and Hilbert sleep labels require experienced human review.
- **Logical exclusion, not deletion:** `BAD_auto_` Annotations exclude downstream analysis windows without deleting samples.
- **Paper parameters remain distinct from engineering choices:** reported PSD/Bandpower parameters are labeled separately from engineering steps such as zero-phase FIR notch filtering.
- **Evidence precedes conclusions:** charts must be interpreted with their CSV/JSON sources, not from filenames or appearance alone.

## Requirements

- Codex or another environment that supports Codex skills
- Python `>=3.11,<3.14`
- MNE `>=1.8`
- NumPy `>=2.0`
- SciPy `>=1.13`
- pandas `>=2.2`
- Matplotlib `>=3.9`
- pytest `>=8` for development and verification
- JupyterLab `>=4` as an optional interactive environment

## Installation

### 1. Install the skill

Run in PowerShell:

```powershell
git clone https://github.com/Chuxingxing701/eeg-one-step-skill.git "$env:USERPROFILE\.codex\skills\eeg-one-step-skill"
```

Restart Codex or begin a new task so it can discover the skill.

### 2. Create the Python environment manually

The skill only detects the environment. It does not run package managers or installers for you. Run these commands yourself from the repository directory:

```powershell
conda env create -f environment.yml
conda activate mouse-eeg
```

Alternatively, follow [`references/environment-setup.md`](references/environment-setup.md) and report the environment name, Python interpreter path, and Conda installation location to Codex.

## Usage

Ask Codex to use the skill, for example:

```text
Use $eeg-one-stop-skill to guide this mouse EEG/EMG analysis.
Start by asking me to select the result export folder. Before opening any EDF,
confirm file order, channels, bipolar formulas, and Markers with me.
```

Or describe a specific goal:

```text
Review 50/100 Hz interference in this N2O mouse EEG dataset and compute Bandpower.
After I approve the smoke test, generate Hilbert candidate sleep stages and an evidence-backed report.
```

The skill first asks for a result export folder and then checks the environment. If experimental information is missing, it stops and asks rather than guessing file order, channel meaning, Markers, or experimental phases.

## Workflow

1. **Select the output directory:** choose where results will be stored before opening EDF/FIF data.
2. **Detect the environment:** report the active interpreter and dependencies; if anything is missing, provide manual setup guidance and stop.
3. **Complete the import checklist:** confirm file order, channel identities and types, bipolar formulas, and every Marker's source and meaning.
4. **Import and reference:** directly concatenate confirmed files, select analysis channels, build bipolar signals, and freeze a checkpoint.
5. **Review artifact candidates:** explain the method, generate every candidate image, review each image against its metrics, and let the user decide whether supported candidates should be excluded.
6. **Review line noise:** compare pre-filter PSD evidence; handle 50 Hz under the approved workflow and wait for the user's explicit 100 Hz decision; then inspect post-filter PSD evidence.
7. **Compute Bandpower:** use user-supplied experimental phases to calculate window-level absolute and relative power while preserving Artifact and boundary windows.
8. **Run Hilbert candidate staging:** explain the exploratory boundary and run a user-selected smoke test; process the target interval only after both confirmations.
9. **Report results:** inspect real PNGs against declared CSV/JSON sources and summarize parameters, decisions, SHA256 evidence, and limitations.

## Outputs And Traceability

Depending on the completed stages, outputs can include:

- Non-overwriting FIF checkpoints and SHA256 hashes
- Input files, complete channel inventories, retained/excluded channels, and software versions
- Marker configuration, Annotations, and user decision records
- Artifact candidate images, metric CSV/JSON files, and completed per-image reviews
- Pre/post-filter PSD, local peak data, and the 50/100 Hz decision
- Window-level Bandpower CSV, parameter JSON, phase comparisons, and time-frequency charts
- Hilbert candidate window CSV, bout CSV, phase summaries, and a manual-review checklist
- Chart source manifests, per-chart interpretation, and a final Markdown report

The exact output depends on which review gates the user has completed and confirmed. Results behind an incomplete gate must remain pending.

## Repository Layout

```text
eeg-one-step-skill/
|-- SKILL.md                         # Skill workflow and review gates
|-- agents/openai.yaml               # Codex display and invocation metadata
|-- scripts/eeg_pipeline/            # Deterministic EEG processing modules
|-- references/
|   |-- environment-setup.md         # Manual environment setup guide
|   `-- chart-analysis-contract.md   # Chart review and conclusion contract
|-- assets/checklist_schema.json     # Input checklist schema
|-- tests/                            # Behavioral, boundary, and regression tests
|-- environment.yml                  # Conda environment definition
`-- pyproject.toml                   # Python package and pytest configuration
```

The repository currently provides reusable Python components and a Codex-guided workflow. It does not yet provide a complete one-command CLI pipeline entry point.

## Development And Verification

After activating an environment with all dependencies, run:

```powershell
python -m pytest -q
```

The suite covers environment detection, input checklists, EDF/FIF import, analysis-channel selection, Markers, bipolar reference, artifact review, line-noise evidence, Bandpower, Hilbert candidate staging, workspace state, and result-reporting boundaries.

## Current Limitations And Research Disclaimer

- This project is intended for research workflows, not clinical diagnosis, treatment decisions, or medical-device use.
- Files are directly concatenated only in the user-confirmed order; this version does not automatically detect or trim overlaps or gaps.
- Artifact candidate detection supports triage and review. It is not automatic artifact diagnosis, and high EMG can represent genuine wake activity.
- Hilbert `Wake_candidate`, `NREM_candidate`, and `REM_candidate` outputs are exploratory candidate labels, not Ground Truth. REM candidates especially require human review.
- Bandpower remains independent of unvalidated sleep candidates; candidate stages must not rewrite or discard Bandpower windows.
- Single-animal EEG changes can describe an electrophysiological trend, but they cannot establish antidepressant efficacy. More animals, replication, behavioral analysis, and experienced human review are still required.

## License

This repository currently does not contain a `LICENSE` file. Unless the author grants separate permission, do not assume that the code is available under an open-source license. Contact the repository owner before copying, modifying, or redistributing it.
