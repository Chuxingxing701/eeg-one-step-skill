"""Create result charts and gate evidence-backed EEG reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .line_noise import StepNotice, _chinese_font
from .models import ChartAnalysis, OverallConclusion
from .workspace import next_versioned_path, write_bytes_exclusive


PRIMARY_BANDS = (
    "Delta",
    "Theta",
    "Sigma",
    "Beta",
    "Low_Gamma",
    "High_Gamma",
)
ADDITIONAL_BAND = "Alpha_additional"
CANDIDATE_STAGES = (
    "Wake_candidate",
    "NREM_candidate",
    "REM_candidate",
    "Uncertain",
    "Artifact",
    "Boundary_Unscored",
)
FORBIDDEN_CHART_LABELS = ("exploratory", "current phase")


@dataclass(frozen=True)
class ReportingInputs:
    bandpower_csv_path: Path
    hilbert_epoch_csv_path: Path
    hilbert_phase_summary_csv_path: Path


@dataclass(frozen=True)
class ResultChartBundle:
    chart_paths: tuple[Path, ...]
    review_manifest_path: Path


@dataclass(frozen=True)
class CompletedChartReviewBundle:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class RunEvidenceSummary:
    marker_summary: str
    artifact_summary: str
    line_noise_summary: str
    reproducibility_summary: str

    def validate(self) -> list[str]:
        fields = {
            "Marker汇总": self.marker_summary,
            "伪迹决定与排除时长": self.artifact_summary,
            "50/100 Hz处理证据": self.line_noise_summary,
            "复现信息": self.reproducibility_summary,
        }
        return [f"{name}不能为空" for name, value in fields.items() if not value.strip()]


@dataclass(frozen=True)
class _ChartSpec:
    title: str
    content_name: str
    chart_kind: str
    source_data_paths: tuple[Path, ...]
    elements: str


def reporting_step_notices() -> tuple[StepNotice, ...]:
    return (
        StepNotice(1, 6, "核对结果输入", "读取Bandpower和Hilbert CSV并核对字段，不推断缺失数据。", "保证每张图有明确同源数据。", "结果输入QC。"),
        StepNotice(2, 6, "生成Bandpower图表", "分别输出论文主频段和Alpha附加指标的时序、阶段箱线图及频段时频图。", "把逐窗数值转换为可检查图像。", "Bandpower图表。"),
        StepNotice(3, 6, "生成Hilbert候选图表", "输出候选分期时间轴和阶段占比，不将候选标签改称正式睡眠分期。", "展示候选状态的时间与阶段分布。", "Hilbert候选图表。"),
        StepNotice(4, 6, "逐图分析", "Codex逐张查看真实PNG并与声明的CSV交叉核对。", "为每张图形成可追溯结论。", "完成的逐图审核。"),
        StepNotice(5, 6, "形成四层结论", "分别判断电生理变化、论文一致性、抗抑郁样趋势和明确抗抑郁作用。", "允许描述单只动物趋势，同时避免把趋势等同于疗效。", "四层总体结论。"),
        StepNotice(6, 6, "输出总报告", "仅在全部图表审核完成后生成图文同步HTML结果页。", "交付可直接阅读且可复查的结果。", "最终结果报告。"),
    )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少{label}：{resolved}")
    return pd.read_csv(resolved)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(missing)}")


def _load_reporting_data(inputs: ReportingInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bandpower = _read_csv(inputs.bandpower_csv_path, "Bandpower逐窗CSV")
    epochs = _read_csv(inputs.hilbert_epoch_csv_path, "Hilbert逐窗CSV")
    summary = _read_csv(inputs.hilbert_phase_summary_csv_path, "Hilbert阶段汇总CSV")
    band_columns = (
        "midpoint_seconds",
        "phase",
        "status",
        *(f"{band}_relative" for band in PRIMARY_BANDS),
        f"{ADDITIONAL_BAND}_relative",
    )
    _require_columns(bandpower, band_columns, "Bandpower逐窗CSV")
    _require_columns(
        epochs,
        ("start_seconds", "stop_seconds", "midpoint_seconds", "phase", "candidate_stage"),
        "Hilbert逐窗CSV",
    )
    _require_columns(
        summary,
        ("phase", "candidate_stage", "duration_seconds", "percentage_of_phase"),
        "Hilbert阶段汇总CSV",
    )
    if bandpower.empty or epochs.empty or summary.empty:
        raise ValueError("报告输入CSV不能为空")
    return bandpower, epochs, summary


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def _phase_order(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(str(value) for value in frame["phase"] if pd.notna(value)))


def masked_relative_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return a plotting series with invalid analysis windows kept as gaps."""

    if "status" not in frame.columns or column not in frame.columns:
        raise ValueError(f"Bandpower时序缺少字段：status或{column}")
    values = frame[column].to_numpy(dtype=float, copy=True)
    values[frame["status"].astype(str).to_numpy() != "Valid"] = np.nan
    return values


def _style_axis(axis: plt.Axes, font) -> None:
    axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.75)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(font)


def _plot_band_timeseries(frame: pd.DataFrame, bands: tuple[str, ...], title: str, destination: Path) -> None:
    if not (frame["status"] == "Valid").any():
        raise ValueError("没有可用于Bandpower制图的Valid窗口")
    font = _chinese_font()
    columns = 2 if len(bands) > 1 else 1
    rows = int(np.ceil(len(bands) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), squeeze=False, constrained_layout=True)
    x_hours = frame["midpoint_seconds"].to_numpy(dtype=float) / 3600.0
    for index, band in enumerate(bands):
        axis = axes.flat[index]
        axis.plot(x_hours, masked_relative_values(frame, f"{band}_relative"), color="#1F4E79", linewidth=1.0)
        axis.set_title(band.replace("_additional", "（附加指标）"), fontproperties=font)
        axis.set_xlabel("记录时间（小时）", fontproperties=font)
        axis.set_ylabel("相对功率", fontproperties=font)
        _style_axis(axis, font)
    for axis in axes.flat[len(bands):]:
        axis.remove()
    figure.suptitle(title, fontproperties=font)
    _save_figure(figure, destination)


def _plot_band_boxplots(frame: pd.DataFrame, bands: tuple[str, ...], title: str, destination: Path) -> None:
    valid = frame.loc[frame["status"] == "Valid"].copy()
    phases = _phase_order(valid)
    if not phases:
        raise ValueError("没有可用于阶段箱线图的Valid窗口")
    font = _chinese_font()
    columns = 2 if len(bands) > 1 else 1
    rows = int(np.ceil(len(bands) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.6 * rows), squeeze=False, constrained_layout=True)
    for index, band in enumerate(bands):
        axis = axes.flat[index]
        values = [
            valid.loc[valid["phase"] == phase, f"{band}_relative"].dropna().to_numpy(dtype=float)
            for phase in phases
        ]
        axis.boxplot(
            values,
            tick_labels=phases,
            patch_artist=True,
            medianprops={"color": "#E67E22", "linewidth": 1.8},
            boxprops={"facecolor": "#DCE6F1", "edgecolor": "#1F4E79"},
            whiskerprops={"color": "#1F4E79"},
            capprops={"color": "#1F4E79"},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none", "markeredgecolor": "#B33A3A"},
        )
        axis.set_title(band.replace("_additional", "（附加指标）"), fontproperties=font)
        axis.set_ylabel("相对功率", fontproperties=font)
        _style_axis(axis, font)
    for axis in axes.flat[len(bands):]:
        axis.remove()
    figure.suptitle(title, fontproperties=font)
    _save_figure(figure, destination)


def _plot_band_heatmap(frame: pd.DataFrame, destination: Path) -> None:
    matrix = np.vstack([frame[f"{band}_relative"].to_numpy(dtype=float) for band in PRIMARY_BANDS])
    matrix[:, frame["status"].to_numpy() != "Valid"] = np.nan
    font = _chinese_font()
    figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_yticks(np.arange(len(PRIMARY_BANDS)), labels=PRIMARY_BANDS, fontproperties=font)
    tick_indices = np.linspace(0, len(frame) - 1, min(7, len(frame)), dtype=int)
    tick_hours = frame.iloc[tick_indices]["midpoint_seconds"].to_numpy(dtype=float) / 3600.0
    axis.set_xticks(tick_indices, labels=[f"{value:.2f}" for value in tick_hours])
    axis.set_xlabel("记录时间（小时）", fontproperties=font)
    axis.set_ylabel("论文主频段", fontproperties=font)
    axis.set_title("EEG论文主频段相对功率时频图", fontproperties=font)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("相对功率", fontproperties=font)
    _style_axis(axis, font)
    _save_figure(figure, destination)


def _plot_hilbert_timeline(frame: pd.DataFrame, destination: Path) -> None:
    font = _chinese_font()
    stage_index = {stage: index for index, stage in enumerate(CANDIDATE_STAGES)}
    unknown = sorted(set(frame["candidate_stage"].astype(str)) - set(stage_index))
    if unknown:
        raise ValueError(f"Hilbert逐窗CSV包含未知候选标签：{', '.join(unknown)}")
    colors = ("#1F4E79", "#2A7F62", "#8E6C8A", "#7F7F7F", "#B33A3A", "#C58B2A")
    figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = frame["midpoint_seconds"].to_numpy(dtype=float) / 3600.0
    for stage, color in zip(CANDIDATE_STAGES, colors):
        mask = frame["candidate_stage"].astype(str) == stage
        axis.scatter(x[mask], np.full(int(mask.sum()), stage_index[stage]), s=8, color=color, label=stage)
    axis.set_yticks(range(len(CANDIDATE_STAGES)), labels=CANDIDATE_STAGES)
    axis.set_xlabel("记录时间（小时）", fontproperties=font)
    axis.set_ylabel("Hilbert候选状态", fontproperties=font)
    axis.set_title("Hilbert候选分期时间轴", fontproperties=font)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _style_axis(axis, font)
    _save_figure(figure, destination)


def _plot_hilbert_phase_proportions(frame: pd.DataFrame, destination: Path) -> None:
    phases = _phase_order(frame)
    font = _chinese_font()
    colors = ("#1F4E79", "#2A7F62", "#8E6C8A", "#7F7F7F", "#B33A3A", "#C58B2A")
    figure, axis = plt.subplots(figsize=(11.5, 6), constrained_layout=True)
    bottom = np.zeros(len(phases), dtype=float)
    for stage, color in zip(CANDIDATE_STAGES, colors):
        values = np.array(
            [
                frame.loc[(frame["phase"] == phase) & (frame["candidate_stage"] == stage), "percentage_of_phase"].sum()
                for phase in phases
            ],
            dtype=float,
        )
        axis.bar(phases, values, bottom=bottom, label=stage, color=color)
        bottom += values
    axis.set_ylabel("阶段占比（%）", fontproperties=font)
    axis.set_title("Hilbert候选分期阶段占比", fontproperties=font)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _style_axis(axis, font)
    _save_figure(figure, destination)


def _write_pending_manifest(specs: list[_ChartSpec], paths: list[Path], destination: Path) -> None:
    charts = []
    for spec, path in zip(specs, paths):
        charts.append(
            {
                "chart_title": spec.title,
                "chart_kind": spec.chart_kind,
                "chart_path": str(path.resolve()),
                "source_data_paths": [str(item.resolve()) for item in spec.source_data_paths],
                "elements": spec.elements,
                "observations": "",
                "numerical_evidence": "",
                "chart_conclusion": "",
                "n2o_relevance": "",
                "limitations": "",
                "review_status": "Pending",
            }
        )
    payload = {
        "review_rule": "Codex必须逐张查看真实PNG并与声明的CSV交叉核对后填写；不得仅凭文件名生成结论。",
        "charts": charts,
    }
    write_bytes_exclusive(destination, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def generate_result_charts(
    inputs: ReportingInputs,
    chart_directory: Path,
    review_directory: Path,
    *,
    now: datetime,
) -> ResultChartBundle:
    bandpower, epochs, summary = _load_reporting_data(inputs)
    inputs = ReportingInputs(
        inputs.bandpower_csv_path.resolve(),
        inputs.hilbert_epoch_csv_path.resolve(),
        inputs.hilbert_phase_summary_csv_path.resolve(),
    )
    box_elements = "橙色线代表中位数；箱体表示Q1至Q3（中间50%数据）；须结合图中须线所示1.5倍四分位距范围和离群点判断分布。"
    specs = [
        _ChartSpec("论文主频段相对功率时序", "论文主频段相对功率时序", "time_series", (inputs.bandpower_csv_path,), "横轴为记录时间，纵轴为相对功率；六个小图分别对应论文主频段。"),
        _ChartSpec("Alpha附加指标相对功率时序", "Alpha附加指标相对功率时序", "time_series", (inputs.bandpower_csv_path,), "横轴为记录时间，纵轴为Alpha附加指标相对功率；该指标不并入论文主频段结论。"),
        _ChartSpec("论文主频段阶段箱线图", "论文主频段阶段箱线图", "boxplot", (inputs.bandpower_csv_path,), box_elements),
        _ChartSpec("Alpha附加指标阶段箱线图", "Alpha附加指标阶段箱线图", "boxplot", (inputs.bandpower_csv_path,), box_elements + " Alpha仅作附加观察。"),
        _ChartSpec("EEG论文主频段相对功率时频图", "EEG论文主频段相对功率时频图", "band_time_heatmap", (inputs.bandpower_csv_path,), "横轴为记录时间，纵轴为论文主频段，颜色表示逐窗相对功率；应查看阶段附近是否出现连续频段增强或减弱，而不是把单个亮点当作效应。"),
        _ChartSpec("Hilbert候选分期时间轴", "Hilbert候选分期时间轴", "candidate_timeline", (inputs.hilbert_epoch_csv_path,), "横轴为记录时间，纵轴为候选状态；每个点来自一个逐窗候选标签，Artifact、Uncertain和Boundary_Unscored保持独立。"),
        _ChartSpec("Hilbert候选分期阶段占比", "Hilbert候选分期阶段占比", "candidate_proportion", (inputs.hilbert_phase_summary_csv_path,), "每根柱表示一个实验阶段，堆叠区段表示各候选状态占该阶段的比例；候选比例不是正式睡眠时长。"),
    ]
    for spec in specs:
        lowered = f"{spec.title} {spec.content_name}".lower()
        if any(token in lowered for token in FORBIDDEN_CHART_LABELS):
            raise ValueError("图表名称只能描述图中内容")
    chart_directory = chart_directory.resolve()
    paths = [next_versioned_path(chart_directory, spec.content_name, ".png", now) for spec in specs]
    _plot_band_timeseries(bandpower, PRIMARY_BANDS, specs[0].title, paths[0])
    _plot_band_timeseries(bandpower, (ADDITIONAL_BAND,), specs[1].title, paths[1])
    _plot_band_boxplots(bandpower, PRIMARY_BANDS, specs[2].title, paths[2])
    _plot_band_boxplots(bandpower, (ADDITIONAL_BAND,), specs[3].title, paths[3])
    _plot_band_heatmap(bandpower, paths[4])
    _plot_hilbert_timeline(epochs, paths[5])
    _plot_hilbert_phase_proportions(summary, paths[6])
    manifest_path = next_versioned_path(review_directory.resolve(), "逐图分析待审核清单", ".json", now)
    _write_pending_manifest(specs, paths, manifest_path)
    return ResultChartBundle(tuple(paths), manifest_path)


def _manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("charts"), list) or not payload["charts"]:
        raise ValueError("逐图分析清单不包含图表")
    return payload


def _validate_analyses(manifest: dict, analyses: tuple[ChartAnalysis, ...]) -> None:
    expected = {str(Path(item["chart_path"]).resolve()): item for item in manifest["charts"]}
    actual = {str(item.chart_path.resolve()): item for item in analyses}
    if expected.keys() != actual.keys() or len(actual) != len(analyses):
        raise ValueError("必须完成全部图表且每张图恰好一份逐图分析")
    for chart_path, analysis in actual.items():
        errors = analysis.validate()
        if errors:
            raise ValueError("；".join(errors))
        if not Path(chart_path).is_file():
            raise FileNotFoundError(chart_path)
        declared = {str(Path(path).resolve()) for path in expected[chart_path]["source_data_paths"]}
        supplied = {str(path.resolve()) for path in analysis.source_data_paths}
        if declared != supplied:
            raise ValueError(f"图表同源数据声明不一致：{chart_path}")
        for source in supplied:
            if not Path(source).is_file():
                raise FileNotFoundError(source)


def write_completed_chart_reviews(
    manifest_path: Path,
    analyses: tuple[ChartAnalysis, ...],
    directory: Path,
    *,
    now: datetime,
) -> CompletedChartReviewBundle:
    manifest = _manifest(manifest_path.resolve())
    _validate_analyses(manifest, analyses)
    by_path = {str(item.chart_path.resolve()): item for item in analyses}
    completed_charts = []
    markdown = ["# 逐图分析完成报告", "", "每张图均须由Codex查看真实PNG并与声明的CSV交叉核对。", ""]
    for item in manifest["charts"]:
        analysis = by_path[str(Path(item["chart_path"]).resolve())]
        completed = dict(item)
        completed.update(
            {
                "elements": analysis.elements,
                "observations": analysis.observations,
                "numerical_evidence": analysis.numerical_evidence,
                "chart_conclusion": analysis.chart_conclusion,
                "n2o_relevance": analysis.n2o_relevance,
                "limitations": analysis.limitations,
                "review_status": "Completed",
            }
        )
        completed_charts.append(completed)
        markdown.extend(
            [
                f"## {item['chart_title']}",
                "",
                f"- 图片：`{item['chart_path']}`",
                f"- 同源数据：{', '.join(f'`{path}`' for path in item['source_data_paths'])}",
                f"- 图表元素：{analysis.elements}",
                f"- 具体观察：{analysis.observations}",
                f"- 数值证据：{analysis.numerical_evidence}",
                f"- 图级结论：{analysis.chart_conclusion}",
                f"- N2O关系：{analysis.n2o_relevance}",
                f"- 局限：{analysis.limitations}",
                "",
            ]
        )
    payload = {
        "source_manifest": str(manifest_path.resolve()),
        "review_status": "Completed",
        "charts": completed_charts,
    }
    json_path = next_versioned_path(directory.resolve(), "逐图分析完成记录", ".json", now)
    markdown_path = next_versioned_path(directory.resolve(), "逐图分析完成报告", ".md", now)
    write_bytes_exclusive(json_path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_bytes_exclusive(markdown_path, "\n".join(markdown).encode("utf-8"))
    return CompletedChartReviewBundle(json_path, markdown_path)


def _relative_image_path(report_path: Path, chart_path: Path) -> str:
    return Path(os.path.relpath(chart_path, report_path.parent)).as_posix()


def render_overall_conclusion_html(
    conclusion: OverallConclusion,
    *,
    single_animal: bool,
    eeg_only: bool,
) -> str:
    errors = conclusion.validate(single_animal=single_animal, eeg_only=eeg_only)
    if errors:
        raise ValueError("；".join(errors))
    boundary = (
        "抗抑郁样趋势只表示单只动物EEG在完整记录的整体观察中，是否呈现与参考研究同向的电生理模式；"
        "这一判断还需要结合更多动物、重复实验和行为学分析，"
        "不能等同于已证实的抗抑郁疗效。"
    )
    return (
        "<section><h2>总体结论</h2>"
        f"<p><strong>电生理变化：</strong>{escape(str(conclusion.electrophysiology_change))}</p>"
        f"<p><strong>与论文方向一致性：</strong>{escape(str(conclusion.paper_consistency))}</p>"
        f"<p><strong>抗抑郁样趋势（单只动物EEG）：</strong>{escape(str(conclusion.antidepressant_like_trend))}</p>"
        f"<p><strong>趋势依据：</strong>{escape(conclusion.trend_rationale)}</p>"
        f"<p><strong>明显抗抑郁作用：</strong>{escape(str(conclusion.antidepressant_effect))}</p>"
        f"<p><strong>总体理由：</strong>{escape(conclusion.rationale)}</p>"
        f"<p><strong>解释边界：</strong>{escape(boundary)}</p>"
        "</section>"
    )


def write_final_result_report(
    completed_reviews: CompletedChartReviewBundle,
    analyses: tuple[ChartAnalysis, ...],
    conclusion: OverallConclusion,
    evidence: RunEvidenceSummary,
    directory: Path,
    *,
    now: datetime,
    single_animal: bool,
    eeg_only: bool,
) -> Path:
    payload = _manifest(completed_reviews.json_path.resolve())
    if payload.get("review_status") != "Completed" or any(
        item.get("review_status") != "Completed" for item in payload["charts"]
    ):
        raise ValueError("全部图表逐图分析完成前不得生成总报告")
    _validate_analyses(payload, analyses)
    errors = evidence.validate()
    if errors:
        raise ValueError("；".join(errors))
    conclusion_html = render_overall_conclusion_html(conclusion, single_animal=single_animal, eeg_only=eeg_only)
    report_path = next_versioned_path(directory.resolve(), "脑电处理结果报告", ".html", now)
    analysis_by_path = {str(item.chart_path.resolve()): item for item in analyses}
    sections = []
    for chart in payload["charts"]:
        analysis = analysis_by_path[str(Path(chart["chart_path"]).resolve())]
        image_path = _relative_image_path(report_path, analysis.chart_path.resolve())
        sections.append(
            "<section>"
            f"<h2>{escape(chart['chart_title'])}</h2>"
            f"<img src=\"{escape(image_path)}\" alt=\"{escape(chart['chart_title'])}\">"
            f"<p><strong>图表元素：</strong>{escape(analysis.elements)}</p>"
            f"<p><strong>具体观察：</strong>{escape(analysis.observations)}</p>"
            f"<p><strong>数值证据：</strong>{escape(analysis.numerical_evidence)}</p>"
            f"<p><strong>图级结论：</strong>{escape(analysis.chart_conclusion)}</p>"
            f"<p><strong>与N2O的关系：</strong>{escape(analysis.n2o_relevance)}</p>"
            f"<p><strong>局限：</strong>{escape(analysis.limitations)}</p>"
            "</section>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>脑电处理结果报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; color: #202124; background: #fff; margin: 0; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 64px; }}
h1, h2 {{ letter-spacing: 0; }}
h1 {{ font-size: 28px; }} h2 {{ font-size: 20px; margin-top: 0; }}
section {{ border-top: 1px solid #d9d9d9; padding: 28px 0; }}
img {{ display: block; max-width: 100%; height: auto; margin: 16px 0 20px; }}
p, li {{ line-height: 1.7; }}
.warning {{ border-left: 4px solid #b33a3a; padding-left: 14px; }}
</style>
</head>
<body><main>
<h1>脑电处理结果报告</h1>
<p class="warning"><strong>结果边界：</strong>这是一个实验性结果，需要人工核查。Hilbert候选标签不是Ground Truth；单只动物EEG可以描述抗抑郁样趋势，但需要整体观察并结合行为学分析，不能单独证明明确疗效。</p>
<section><h2>运行证据</h2>
<p><strong>Marker：</strong>{escape(evidence.marker_summary)}</p>
<p><strong>伪迹：</strong>{escape(evidence.artifact_summary)}</p>
<p><strong>工频处理：</strong>{escape(evidence.line_noise_summary)}</p>
<p><strong>复现信息：</strong>{escape(evidence.reproducibility_summary)}</p></section>
{''.join(sections)}
{conclusion_html}
</main></body></html>
"""
    write_bytes_exclusive(report_path, html.encode("utf-8"))
    return report_path
