---
name: eeg-one-stop-skill
description: Guide and run a reproducible mouse EEG workflow from EDF/FIF import through direct concatenation, user-supplied Markers, bipolar reference, artifact review, line-noise filtering, paper-aligned Bandpower, Hilbert EEG+EMG candidate sleep staging, and evidence-backed chart reporting. Use this skill whenever a user asks to preprocess mouse EEG/EMG, analyze N2O EEG, build Bandpower or time-frequency outputs, review 50/100 Hz interference, or produce Hilbert Wake/NREM/REM candidate results, even when they do not explicitly request a pipeline.
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
cited paper, whether the complete single-animal record presents an
antidepressant-like electrophysiological trend, and whether it supports an
obvious antidepressant effect. The trend may be `支持`, `不支持`, or `证据不足`,
but it must have a separate rationale and state that more animals and
behavioral analysis are still required. A single-animal EEG-only run cannot
use that trend to claim confirmed efficacy; the obvious antidepressant-effect
verdict remains `证据不足`.


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

Bandpower values are independent of candidate sleep-stage labels. Do not use
unvalidated Hilbert labels to rewrite or discard Bandpower windows.


## Hilbert Candidate Staging Gate

Render every item from `hilbert_step_notices()` before the matching action.
Do not run the full target interval until both user confirmations are recorded.

1. **步骤 1/8：确认Hilbert输入。** 核对滤波冻结版、Bandpower、
   `EEG_bipolar`、`EMG_bipolar`、Marker、BAD和SHA256；二者的窗口和
   时间轴必须一致。
2. **步骤 2/8：说明探索性边界。** 先完整展示
   `hilbert_method_explanation()`。明确40个采样点中值滤波、Hilbert包络
   不会去除心电、标签不是Ground Truth，并等待用户接受。
3. **步骤 3/8：运行Smoke Test。** 短窗口必须由用户提供，不得自动选择。
   调用`write_hilbert_smoke_evidence()`输出EEG、EMG、Hilbert包络和BAD
   背景图；图名直接写窗口内容，不加exploratory或current phase。
4. **步骤 4/8：确认Smoke Test。** 让用户查看全部图片；在用户明确确认前，
   `smoke_test_accepted`保持false，不得调用
   `run_hilbert_candidate_staging()`运行目标时段。
5. **步骤 5/8：提取Hilbert特征。** 使用`scipy.signal.hilbert`计算EMG
   绝对包络并进行40个采样点中值滤波；固定众数加2SD阈值保持关闭。
6. **步骤 6/8：校准并生成候选。** 只在用户提供的校准区间内排除无效窗后
   计算Q25/Q50/Q75。EEG Bandpower为主、Hilbert EMG为辅，输出
   Wake_candidate、NREM_candidate和REM_candidate及规则命中证据。
7. **步骤 7/8：保留无效与冲突。** BAD必须写Artifact；阶段边界必须写
   Boundary_Unscored；缺失、冲突或低置信必须写Uncertain，不得虚构概率
   或强制赋为Wake/NREM/REM。
8. **步骤 8/8：保存Bout与证据。** 用相邻窗口中心的中点形成非重叠时间箱，
   只合并同阶段、同标签且时间连续的Bout，不删除短Bout。保存逐窗CSV、
   Bout CSV、阶段汇总、配置和报告，并生成待人工复核清单。

Treat all candidate durations and phase proportions as experimental engineering
outputs. Require experienced human review, especially for REM_candidate, before
using any label in biological interpretation.


## Result Reporting Gate

Read `references/chart-analysis-contract.md` in full before this gate. Render
every item from `reporting_step_notices()` before the matching action so the
user sees the current step, method, purpose, and output.

1. **步骤 1/6：核对结果输入。** 读取Bandpower逐窗CSV、Hilbert候选逐窗CSV
   和Hilbert阶段汇总CSV。核对必需字段、有效窗口状态和同源文件路径；缺少字段
   时停止，不从文件名或图形外观补值。
2. **步骤 2/6：生成Bandpower图表。** 调用`generate_result_charts()`分别
   输出论文主频段相对功率时序、Alpha附加指标时序、阶段箱线图和论文主频段
   相对功率时频图。图名必须直接说明图中内容，严禁加入`exploratory`、
   `current phase`或其他工作流状态词。Alpha始终与论文主频段分开。
3. **步骤 3/6：生成Hilbert候选图表。** 输出Hilbert候选分期时间轴和阶段
   占比。不得把Wake_candidate、NREM_candidate或REM_candidate改称正式
   睡眠分期；Artifact、Uncertain与Boundary_Unscored必须保留。
4. **步骤 4/6：逐图分析。** 对清单中的每张真实PNG调用`view_image`逐张
   查看，并读取`source_data_paths`声明的同源CSV。不得仅凭文件名、自动摘要
   或预期方向写结论。每张图必须填写图表元素、具体观察、数值证据、图级结论、
   与N2O的关系、局限和`review_status`。箱线图必须解释：橙色线代表中位数，
   箱体代表Q1至Q3，须线通常延伸至1.5倍四分位距范围内的最远值，范围外点为
   离群点；然后结合各阶段中位数、离散程度和重叠程度说明是否存在清晰差异。
   时频图必须说明横轴、纵轴和颜色，并检查阶段附近是否存在连续频段变化；
   单个亮点不能单独作为N2O效应证据。
5. **步骤 5/6：形成四层结论。** 逐图审核全部完成后调用
   `write_completed_chart_reviews()`。总体结论必须分别回答电生理变化、
   与论文方向一致性、单只动物是否呈现抗抑郁样趋势、是否支持明显抗抑郁作用，
   并分别给出理由。趋势必须根据完整记录的整体观察判断，可以写`支持`、`不支持`
   或`证据不足`；同时必须说明还需要更多动物、重复实验和行为学分析，不能等同
   于已证实疗效。单只动物EEG的明显抗抑郁作用仍必须写`证据不足`，不能用频段
   变化替代行为学或群体证据。
6. **步骤 6/6：输出总报告。** 只有所有图表`review_status`均为Completed
   时才可调用`write_final_result_report()`。结果页必须让图片和对应解释相邻，
   并汇总Marker、伪迹决定与排除时长、50/100 Hz处理决定、前后PSD证据、
   软件版本和SHA256。报告结论必须明确写“这是一个实验性结果，需要人工核查”；
   该警告不得被放入图名而妨碍图片直接使用。

Chart drawing is deterministic, but chart interpretation is not automatic.
When real output charts do not exist, leave the review pending and do not
fabricate a biological conclusion from synthetic test data.
