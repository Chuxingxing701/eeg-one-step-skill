---
name: eeg-one-stop-skill
description: Guide and run a reproducible mouse EEG/EMG workflow from EDF/FIF import through direct concatenation, user-supplied Markers, bipolar reference, artifact review, line-noise filtering, paper-aligned Bandpower, pinned Antila/PySleep staging, and evidence-backed Chinese reporting. Use this skill whenever a user asks to preprocess mouse EEG/EMG, analyze N2O EEG, compute Bandpower or time-frequency outputs, review 50/100 Hz interference, stage Wake/NREM/REM, or prepare traceable data for journal figures.
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
- 原始EDF或FIF不得修改。直接拼接FIF只保留双极公式涉及的用户确认分析通道，
  通道顺序按源文件头保留并应用用户确认的EEG/EMG类型；未进入双极公式的通道
  不得静默混入分析文件。拼接报告必须记录保留通道、EEG/EMG类型和排除通道，源文件完整通道清单及SHA256仍保留
  在输入证据中。
- Preserve an immutable checkpoint before each destructive-looking decision.
  BAD annotations exclude analysis windows but never delete samples.
- Pause for user confirmation at Marker, artifact, 100 Hz, and Antila source/calibration gates.
- Keep `Artifact`, `Uncertain`, and `Boundary_Unscored` labels unchanged.
- Use only the pinned Antila/PySleep method for sleep staging; do not add alternative staging branches.
- In user-facing text, call outputs `Antila分期结果`, not candidate staging.
  Do not name, compare, recommend, or fall back to retired staging methods.
- Every result handoff must point to the data inventory and journal-redraw guide,
  even when the current request focuses on only one analysis stage.

## Reporting Contract

Before producing the final report, read
`references/chart-analysis-contract.md`. Inspect every generated chart with
Codex image analysis and cross-check it against its declared CSV/JSON source.
Explain every chart beside the image and keep chart names limited to the
content shown. Put experimental and manual-review warnings in conclusions,
not in chart titles.

The overall conclusion must directly answer the current design's research
questions: acute gas-period effects, recovery 0-2 h effects, longer-term
persistence, sleep-structure change, and paper-parameter consistency. Each
answer must be `成立`, `部分成立`, `不成立`, or `本设计不可计算`, with numerical
evidence. Do not append generic sample-size or future-validation boilerplate.
When no matched Control exists, mark N2O-Control questions as
`本设计不可计算` and report only within-record time-related changes.


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


## Line Noise Gate

Before this gate, show the full output of `line_noise_method_explanation()`.
For every action, render the matching `StepNotice` first so the user always
sees the current step number, method, purpose, and expected output.

1. **步骤 1/7：输入结构复核。** 方法：检查样本数、通道、通道类型、采样率
   和Annotations，不修改信号。输入不通过时立即停止。
2. **步骤 2/7：计算滤波前PSD。** 方法：调用 `compute_paper_psd()`，采用
   论文报告的3秒Hann-Welch窗口、50%重叠，并以
   `reject_by_annotation=True`排除BAD窗口。EEG和EMG必须分别显式选择。
3. **步骤 3/7：查看滤波前工频图。** 方法：生成完整PSD、45至55 Hz及
   95至105 Hz三张图；使用`view_image`查看每一张图片，并与PSD和局部峰CSV
   交叉核对。说明图表含义、窄峰证据及局限。
4. **步骤 4/7：确认100 Hz处理决定。** 方法：50 Hz按已批准要求滤除；
   100 Hz必须向用户展示图片和数值证据，并取得用户明确决定。将决定记录为
   `filter_100=True`或`filter_100=False`；缺少决定时必须停止。
5. **步骤 5/7：执行工频陷波。** 方法：调用`apply_line_noise_filter()`。
   这是MNE离线零相位FIR工程处理，不是论文报告的滤波参数。严禁静默增加
   0.5-100 Hz带通或其他频率。
6. **步骤 6/7：计算滤波后PSD并对照。** 方法：继续使用相同的论文PSD几何，
   再生成同样三张图并逐图检查。确认目标窄峰降低，同时检查邻近频率、样本数、
   通道和Annotations未发生非预期变化。
7. **步骤 7/7：冻结滤波结果。** 方法：将结果保存到
   `数据冻结/06_滤波后`，不得覆盖；同时保存Annotations、参数、软件版本、
   用户100 Hz决定、前后PSD证据和SHA256。

The paper source applies to the 3-second Hann-Welch PSD with 50% overlap.
The paper did not report the notch family, phase, order, width, or a
0.5-100 Hz bandpass. The zero-phase FIR notch is therefore labeled
`工程处理` everywhere. No automatic 100 Hz verdict is allowed.


## Bandpower Gate

Render every item from `bandpower_step_notices()` before the matching action.
Each prompt must show the step number, method, purpose, and output.

1. **步骤 1/7：确认Bandpower输入。** 核对滤波冻结版、EEG通道、采样率、
   Marker、BAD和SHA256，不修改数据。
2. **步骤 2/7：确认实验阶段区间。** Baseline、Treatment、Recovery等区间
   必须由用户提供；程序不得推断名称、起止时间或时长。
3. **步骤 3/7：声明频段与窗参数。** 论文主频段为Delta、Theta、Sigma、
   Beta、Low Gamma和High Gamma；采用5秒窗、2.5秒步长及
   2秒Hann-Welch子窗、50%重叠。Alpha仅为附加指标。
4. **步骤 4/7：逐窗计算Bandpower。** 调用`compute_bandpower()`，输出每窗
   绝对功率和相对于0.5至100 Hz总功率的相对值。
5. **步骤 5/7：处理无效窗口。** 任何BAD重叠标为Artifact；跨阶段窗口标为
   Boundary_Unscored；这些窗口的功率必须写NaN，不得写0或强制分配阶段。
6. **步骤 6/7：保存Bandpower证据。** 输出逐窗CSV、参数JSON和方法报告，
   不得覆盖已有版本；主频段与Alpha附加指标保持分离。
7. **步骤 7/7：进入图表与结论分析。** 使用主频段制作时序、阶段比较和
   时频图；Alpha单独展示。每张图必须由Codex查看并结合CSV解释。

Bandpower values are independent of Antila sleep-stage labels. Do not use
sleep stages to rewrite or discard Bandpower windows.


## Antila Staging Gate

Render every item from `antila_step_notices()` before the matching action.
Do not run staging until source provenance and calibration scope are recorded.

1. **步骤 1/8：确认Antila输入。** 核对滤波冻结FIF、Bandpower窗口轴、
   `EEG_bipolar`、`EMG_bipolar`、Marker、status和SHA256；窗口必须严格同轴。
2. **步骤 2/8：确认作者源码。** 用户提供本地`tortugar/Lab` checkout。
   调用`validate_antila_source()`确认commit
   `bcb8dae1594e64a511545e34f6050e2a417c1f45`和`sleepy.py` SHA256。
   缺失或不匹配时停止；不得静默clone、下载或替换源码。
3. **步骤 3/8：确认校准范围。** 默认只用当前记录的Valid窗传给作者
   `use_idx`。仅当Control/N2O确属确认配对记录且用户批准时，才允许
   `pooled_valid`校准，并标记为项目扩展。
4. **步骤 4/8：准备作者输入。** 版本化输出EEG.mat、EMG.mat、info.txt及
   作者频谱文件；记录每个文件的结构、大小和SHA256，不覆盖已有文件。
5. **步骤 5/8：运行作者方法。** 调用固定版本`calculate_spectrum()`和
   未修改的`sleep_state()`；保存完整参数、运行记录和作者detail keys。
6. **步骤 6/8：对齐项目主轴。** 用`align_antila_states()`把作者输出严格
   对齐5秒窗/2.5秒步长项目轴。作者额外零填充尾窗单独保存，不强行合并。
7. **步骤 7/8：恢复保护状态。** Valid窗可写Wake、NREM、REM或Uncertain；
   Artifact和Boundary_Unscored覆盖作者标签；不得改写Bandpower值。
8. **步骤 8/8：保存睡眠结构证据。** 输出逐窗分期、阶段时长、睡眠片段、
   状态转换、校准配置、源码清单、方法报告、开发日志和SHA256。

The staging output is the pinned Antila result for this pipeline. Preserve
method provenance and invalid states so later figures can be regenerated from
the CSV rather than inferred from report images.


## Result Reporting Gate

Read `references/chart-analysis-contract.md` in full before this gate. Render
every item from `reporting_step_notices()` before the matching action so the
user sees the current step, method, purpose, and output.

1. **步骤 1/6：核对结果输入。** 读取Bandpower逐窗CSV、Antila逐窗CSV
   和Antila阶段汇总CSV。核对必需字段、有效窗口状态和同源文件路径；缺少字段
   时停止，不从文件名或图形外观补值。
2. **步骤 2/6：生成Bandpower图表。** 调用`generate_result_charts()`分别
   输出论文主频段相对功率时序、Alpha附加指标时序、阶段箱线图和论文主频段
   相对功率时频图。图名必须直接说明图中内容，严禁加入`exploratory`、
   `current phase`或其他工作流状态词。Alpha始终与论文主频段分开。
3. **步骤 3/6：生成Antila分期图表。** 输出Antila睡眠分期时间轴和阶段
   占比；图内使用中文状态名称，CSV保留Wake、NREM、REM枚举；Artifact、
   Uncertain与Boundary_Unscored必须保留。
4. **步骤 4/6：逐图分析。** 对清单中的每张真实PNG调用`view_image`逐张
   查看，并读取`source_data_paths`声明的同源CSV。不得仅凭文件名、自动摘要
   或预期方向写结论。每张图必须填写图表元素、具体观察、数值证据、图级结论、
   与N2O的关系、局限和`review_status`。箱线图必须解释：橙色线代表中位数，
   箱体代表Q1至Q3，须线通常延伸至1.5倍四分位距范围内的最远值，范围外点为
   离群点；然后结合各阶段中位数、离散程度和重叠程度说明是否存在清晰差异。
   时频图必须说明横轴、纵轴和颜色，并检查阶段附近是否存在连续频段变化；
   单个亮点不能单独作为N2O效应证据。
5. **步骤 5/6：回答研究问题。** 逐图审核全部完成后调用
   `write_completed_chart_reviews()`。用`ResearchQuestionConclusions`分别回答
   通气期急性效应、恢复0–2小时、长期持续性、睡眠结构变化和论文参数方向。
   结论只能写`成立`、`部分成立`、`不成立`或`本设计不可计算`，每项附数值依据。
6. **步骤 6/6：输出总报告。** 只有所有图表`review_status`均为Completed
   时才可调用`write_final_result_report()`。结果页必须让图片和对应解释相邻，
   并汇总Marker、伪迹决定与排除时长、50/100 Hz处理决定、前后PSD证据、
   软件版本和SHA256。调用`write_data_inventory()`输出数据获取清单，并调用
   `write_journal_redraw_guide()`输出期刊重绘说明，覆盖EDF/FIF、Antila、
   Bandpower、图表源CSV、字段、单位与筛选方法；报告PNG只用于汇报，
   期刊图从CSV重绘。

Chart drawing is deterministic, but chart interpretation is not automatic.
When real output charts do not exist, leave the review pending and do not
fabricate a biological conclusion from synthetic test data.
