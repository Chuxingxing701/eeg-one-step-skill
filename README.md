# EEG One-Stop Skill

面向 Codex 的可复现小鼠 EEG/EMG 工作流：从 EDF/FIF 导入、Marker、双极参考、伪迹审核和工频处理，直到论文频段 Bandpower、固定版本 Antila/PySleep 睡眠分期、中文图表、数据清单与研究问题结论。

[English](#english)

## 核心能力

- 按用户确认的文件顺序直接拼接 EDF/FIF，不自动裁剪重叠或缺口。
- 只保留双极公式涉及的分析通道，并记录通道类型、排除通道和 SHA256。
- 将用户提供的 Marker 转换到统一样本时间轴，不猜测实验阶段。
- 生成伪迹候选并逐图审核；BAD 只逻辑排除，不删除样本。
- 分别检查 EEG/EMG 的 50 Hz 和 100 Hz 证据；100 Hz 处理需用户决定。
- 使用5秒窗、2.5秒步长、2秒 Hann-Welch 子窗计算绝对/相对 Bandpower。
- 只使用固定版本 Antila/PySleep 输出 Wake、NREM、REM、Uncertain、Artifact 和 Boundary_Unscored。
- 生成睡眠状态构成、睡眠片段、状态转换和状态内频段功率。
- 图表可见文字使用中文，并逐张与同源 CSV/JSON 核对。
- 输出数据获取清单和期刊重绘说明，让后续人员从标准化 CSV 重新绘图。
- 直接回答急性效应、恢复0–2小时、长期持续性、睡眠结构和论文参数方向。

## Antila 来源

本 skill 不复制或再分发上游作者源码。运行睡眠分期时，用户需要提供本地 checkout：

- Repository: https://github.com/tortugar/Lab
- Commit: bcb8dae1594e64a511545e34f6050e2a417c1f45
- PySleep/sleepy.py SHA256: da7ad9b5a771362cc529cca98a19879837337d4e40969f684067bcacc0148a04

skill 会在产生任何分期输出前验证路径、commit 和 SHA256。验证失败时停止，不静默 clone、下载或替换文件。

## 工作流程

1. 选择输出目录。
2. 检测 Python、MNE、NumPy、SciPy、pandas、Matplotlib、pytest 和 Git。
3. 确认文件顺序、同一动物连续记录、通道身份、双极公式和全部 Marker。
4. 直接拼接并冻结原始分析通道。
5. 建立 EEG/EMG 双极参考并完成数值 QC。
6. 生成并审核全部伪迹候选，只排除有证据支持的片段。
7. 检查 50/100 Hz，应用用户批准的工程陷波并复核。
8. 计算独立 Bandpower 主表；无效窗口保持 NaN。
9. 验证 Antila 作者源码，准备 MAT/频谱输入并运行固定版本作者方法。
10. 将作者输出对齐项目5秒窗轴，额外零填充尾窗单独保存。
11. 恢复 Artifact 和 Boundary_Unscored，生成阶段时长、睡眠片段和转换。
12. 生成中文图表，逐图核对同源 CSV。
13. 输出研究问题结论、数据清单、期刊重绘说明、开发日志和 SHA256。

## 校准

默认使用当前记录的 Valid 窗作为作者 use_idx。只有 Control/N2O 确认为配对记录且用户明确批准时，才允许 pooled Valid 校准；该模式必须标记为项目扩展。

## 结论格式

每项研究问题使用以下判断之一：

- 成立
- 部分成立
- 不成立
- 本设计不可计算

没有匹配 Control 时，N2O-Control 问题写“本设计不可计算”，同时可以报告通气前、通气中和通气后的记录内部变化。报告不会自动追加与当前结果无关的样本量模板。

## 主要输出

- 不可覆盖的拼接、双极、伪迹和滤波 FIF 检查点
- 输入 manifest、Marker、Annotations、软件版本和 SHA256
- 伪迹候选图、指标和逐图审核记录
- 50/100 Hz 前后 PSD 和用户决定
- Bandpower 逐窗 CSV、配置和方法报告
- Antila 逐窗分期、作者尾窗、阶段汇总、睡眠片段、状态转换和运行记录
- 中文结果图及逐图审核
- 数据获取清单 CSV
- 数据获取与期刊重绘说明 Markdown
- 直接回答研究问题的最终 HTML 报告

## 环境

- Python >=3.11,<3.14
- MNE >=1.8
- NumPy >=2.0
- SciPy >=1.13
- pandas >=2.2
- Matplotlib >=3.9
- pytest >=8
- Git
- JupyterLab >=4（可选）

本 skill 只检测环境，不运行包管理器或安装器。

## 安装

PowerShell命令：

    git clone https://github.com/Chuxingxing701/eeg-one-step-skill.git "$env:USERPROFILE\.codex\skills\eeg-one-step-skill"

重启 Codex 或开始新任务。准备 Antila 上游源码时由用户手动 clone 并 checkout 固定 commit。

## 使用示例

    使用 $eeg-one-stop-skill 处理这批小鼠 EEG/EMG。
    从选择输出目录开始，确认文件顺序、通道、双极公式和 Marker。
    完成伪迹与工频审核、Bandpower和固定版本Antila分期。
    图表全部使用中文，并输出数据清单和期刊重绘说明。

## 开发验证

    python -m pytest -q

发布前还必须运行系统 skill validator 和退役方法扫描测试。

## 许可说明

本仓库当前未包含 LICENSE。Antila上游仓库同样未显示 LICENSE，因此本项目只记录来源并调用用户本地 checkout，不复制上游 sleepy.py。

---

<a id="english"></a>

## English

EEG One-Stop Skill is a reproducible mouse EEG/EMG workflow for Codex. It covers EDF/FIF import, user-supplied Markers, bipolar references, artifact review, line-noise processing, paper-aligned Bandpower, pinned Antila/PySleep sleep staging, Chinese report charts, data inventories, and direct research-question conclusions.

## Capabilities

- Concatenate files only in the user-confirmed order.
- Preserve source EDF/FIF and freeze versioned checkpoints.
- Review artifact candidates without deleting samples.
- Review and process 50/100 Hz line noise with explicit user decisions.
- Compute 5-second / 2.5-second-step Bandpower with 2-second Hann-Welch subwindows.
- Use only pinned Antila/PySleep staging.
- Preserve Wake, NREM, REM, Uncertain, Artifact, and Boundary_Unscored.
- Produce Chinese-visible chart labels and source-linked chart reviews.
- Write a data inventory and journal-redraw guide.
- Answer acute, recovery, long-term, sleep-structure, and paper-parameter questions directly.

## Pinned Antila Source

Users provide a local checkout of https://github.com/tortugar/Lab at commit bcb8dae1594e64a511545e34f6050e2a417c1f45. The required PySleep/sleepy.py SHA256 is da7ad9b5a771362cc529cca98a19879837337d4e40969f684067bcacc0148a04.

The skill validates source provenance before producing staging outputs. It never downloads or silently replaces the author source.

## Workflow

1. Select an output directory.
2. Detect the Python and Git environment.
3. Confirm files, channels, bipolar formulas, and Markers.
4. Concatenate and freeze selected channels.
5. Build and verify bipolar EEG/EMG.
6. Review artifact candidates.
7. Review 50/100 Hz evidence and apply approved engineering filters.
8. Compute independent Bandpower tables.
9. Validate and run pinned Antila/PySleep.
10. Align author windows to the project axis and preserve author-only tail windows separately.
11. Restore invalid states and summarize sleep structure.
12. Generate Chinese charts and review every chart against source data.
13. Write research-question conclusions, a data inventory, a redraw guide, logs, and hashes.

## Conclusion Values

Each research question is reported as one of:

- 成立
- 部分成立
- 不成立
- 本设计不可计算

When no matched Control exists, N2O-Control effects are not computed. Within-record temporal changes can still be reported.

## Installation

PowerShell:

    git clone https://github.com/Chuxingxing701/eeg-one-step-skill.git "$env:USERPROFILE\.codex\skills\eeg-one-step-skill"

Restart Codex after installation. Users prepare the pinned Antila checkout manually.

## Verification

Run the full pytest suite and the system skill validator before release.

## Licensing

This repository currently has no LICENSE. The upstream Antila repository also does not display a license, so this project records provenance and calls a user-provided local checkout rather than redistributing sleepy.py.
