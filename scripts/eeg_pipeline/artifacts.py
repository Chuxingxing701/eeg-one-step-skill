"""Detect, explain, review, and annotate mouse EEG/EMG artifact candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
import csv
import io
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from .workspace import next_versioned_path, write_bytes_exclusive


ArtifactVerdict = Literal["支持伪迹", "不支持伪迹", "无法判断"]
ArtifactBatchDecision = Literal["exclude_supported", "retain_all"]


@dataclass(frozen=True)
class ArtifactDetectionConfig:
    window_seconds: float = 5.0
    robust_z_threshold: float = 6.0
    context_seconds: float = 10.0
    channel_names: tuple[str, ...] = ("EEG_bipolar", "EMG_bipolar")


@dataclass(frozen=True)
class ArtifactCandidate:
    candidate_id: str
    onset_seconds: float
    duration_seconds: float
    possible_types: tuple[str, ...]
    triggered_metrics: tuple[str, ...]
    algorithm_reason: str


@dataclass(frozen=True)
class ArtifactDetectionResult:
    config: ArtifactDetectionConfig
    candidates: tuple[ArtifactCandidate, ...]
    window_count: int
    skipped_existing_bad_count: int


@dataclass(frozen=True)
class ArtifactReview:
    candidate_id: str
    verdict: ArtifactVerdict
    artifact_type: str
    reason: str
    image_evidence: str


@dataclass(frozen=True)
class ArtifactReviewBundle:
    method_path: Path
    candidates_csv_path: Path
    review_manifest_path: Path
    review_markdown_path: Path
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ArtifactDecisionResult:
    raw: mne.io.BaseRaw
    decision: ArtifactBatchDecision
    bad_count: int
    bad_duration_seconds: float


def _validate_config(config: ArtifactDetectionConfig) -> None:
    if config.window_seconds <= 0:
        raise ValueError("伪迹检测窗口必须大于0秒")
    if config.robust_z_threshold <= 0:
        raise ValueError("稳健异常阈值必须大于0")
    if config.context_seconds < 0:
        raise ValueError("图片上下文时长不能为负数")
    if not config.channel_names:
        raise ValueError("必须指定至少一个伪迹检测通道")
    if len(config.channel_names) != len(set(config.channel_names)):
        raise ValueError("伪迹检测通道不能重复")


def artifact_method_explanation(config: ArtifactDetectionConfig) -> str:
    _validate_config(config)
    return (
        "# 伪迹候选检测方法与人工判断说明\n\n"
        "## 本阶段做什么\n\n"
        f"程序按{config.window_seconds:g}秒窗口读取用户确认的双极EEG和EMG，"
        "分别计算峰峰值、RMS（均方根幅度）和相邻采样点变化的RMS。"
        "每项指标使用本次记录自身的中位数和MAD（中位绝对偏差）建立稳健基线；"
        f"稳健异常分数大于{config.robust_z_threshold:g}，或存在NaN、Inf、"
        "完全平直线时，只进入候选清单。程序不会删除样本，也不会在用户确认前"
        "把候选强制写成BAD。既有BAD窗口不会重复检测。\n\n"
        "## 常见伪迹怎样判断\n\n"
        "- 高幅瞬变或电极弹跳：波形突然出现孤立、陡峭的大幅偏转，随后快速恢复；"
        "相邻采样点变化通常异常增大。\n"
        "- 运动或电缆扰动：EEG与EMG在同一时段同步出现大幅、不规则变化，"
        "有时伴随基线漂移；同步性支持机械扰动，但不能单凭同步性定案。\n"
        "- 肌电污染：EEG上出现密集高频活动，同时EMG明显增强；"
        "清醒或真实运动也可产生相同表现，因此高EMG不能单独证明伪迹。\n"
        "- 信号中断或平直线：一个或多个通道持续接近常数、重复为零，"
        "且缺少正常生理波动。\n"
        "- 削顶或饱和：多个峰值反复贴在完全相同的上限或下限，波峰呈平顶；"
        "这主要依靠逐图检查确认。\n"
        "- 工频干扰：规则、持续的周期波及PSD中50 Hz或100 Hz尖峰。"
        "它不在本阶段按片段删除，而在工频检查与滤波阶段单独处理。\n\n"
        "## 每张图必须回答\n\n"
        "Codex必须查看每张候选图并写明：是否支持伪迹、可能是哪一种伪迹、"
        "判断理由，以及图中哪些波形特征支持该结论。无法区分真实活动与伪迹时，"
        "必须写“无法判断”，不得强行排除。\n"
    )


def _bad_intervals(raw: mne.io.BaseRaw) -> tuple[tuple[float, float], ...]:
    intervals = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if str(description).upper().startswith("BAD") and duration > 0:
            relative_onset = float(onset) - float(raw.first_time)
            intervals.append((relative_onset, relative_onset + float(duration)))
    return tuple(intervals)


def _overlaps_any(
    start: float,
    stop: float,
    intervals: tuple[tuple[float, float], ...],
) -> bool:
    return any(start < bad_stop and stop > bad_start for bad_start, bad_stop in intervals)


def _robust_upper_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    usable = values[valid & np.isfinite(values)]
    if usable.size == 0:
        return result
    median = float(np.median(usable))
    mad = float(np.median(np.abs(usable - median)))
    usable_mask = valid & np.isfinite(values)
    if mad <= np.finfo(float).eps:
        result[usable_mask] = np.where(values[usable_mask] > median, np.inf, 0.0)
    else:
        result[usable_mask] = 0.6744897501960817 * (
            values[usable_mask] - median
        ) / mad
    return result


def _candidate_types_and_reason(
    channel_names: tuple[str, ...],
    trigger_details: list[str],
    high_channels: set[str],
    derivative_channels: set[str],
    flatline_channels: set[str],
    nonfinite_channels: set[str],
) -> tuple[tuple[str, ...], str]:
    possible_types: list[str] = []
    if nonfinite_channels:
        possible_types.append("缺失值或非有限信号候选")
    if flatline_channels:
        possible_types.append("信号中断或平直线候选")
    eeg_high = any("EEG" in name.upper() for name in high_channels)
    emg_high = any("EMG" in name.upper() for name in high_channels)
    if eeg_high and emg_high:
        possible_types.append("运动或电缆扰动候选")
    elif eeg_high:
        possible_types.append("EEG高幅瞬变候选")
    elif emg_high:
        possible_types.append("EMG高活动候选（也可能是真实活动）")
    if derivative_channels:
        possible_types.append("快速跳变或电极弹跳候选")
    if not possible_types:
        possible_types.append("未分类异常候选")
    ordered_types = tuple(dict.fromkeys(possible_types))
    reason = (
        f"窗口在{'、'.join(channel_names)}的自动指标中触发候选条件："
        + "；".join(trigger_details)
        + "。这些指标只说明波形偏离本记录的稳健基线，仍需逐图排除真实活动。"
    )
    return ordered_types, reason


def detect_artifact_candidates(
    raw: mne.io.BaseRaw,
    config: ArtifactDetectionConfig,
) -> ArtifactDetectionResult:
    """Detect candidates without modifying samples or annotations."""

    _validate_config(config)
    missing = [name for name in config.channel_names if name not in raw.ch_names]
    if missing:
        raise ValueError(f"缺少伪迹检测通道：{', '.join(missing)}")
    sfreq = float(raw.info["sfreq"])
    samples_per_window = int(round(config.window_seconds * sfreq))
    if samples_per_window <= 0:
        raise ValueError("伪迹检测窗口换算后的样本数必须大于0")
    window_count = math.ceil(raw.n_times / samples_per_window)
    channel_count = len(config.channel_names)
    metric_names = ("peak_to_peak", "rms", "derivative_rms")
    metrics = np.full((window_count, channel_count, len(metric_names)), np.nan)
    flatline = np.zeros((window_count, channel_count), dtype=bool)
    nonfinite = np.zeros((window_count, channel_count), dtype=bool)
    skipped = np.zeros(window_count, dtype=bool)
    bad_intervals = _bad_intervals(raw)

    for window_index in range(window_count):
        start_sample = window_index * samples_per_window
        stop_sample = min(start_sample + samples_per_window, raw.n_times)
        start_seconds = start_sample / sfreq
        stop_seconds = stop_sample / sfreq
        if _overlaps_any(start_seconds, stop_seconds, bad_intervals):
            skipped[window_index] = True
            continue
        data = raw.get_data(
            picks=list(config.channel_names),
            start=start_sample,
            stop=stop_sample,
            reject_by_annotation=None,
        )
        for channel_index in range(channel_count):
            values = data[channel_index]
            is_finite = bool(np.all(np.isfinite(values)))
            nonfinite[window_index, channel_index] = not is_finite
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            peak_to_peak = float(np.ptp(finite_values))
            rms = float(np.sqrt(np.mean(np.square(finite_values))))
            differences = np.diff(finite_values)
            derivative_rms = (
                float(np.sqrt(np.mean(np.square(differences))))
                if differences.size
                else 0.0
            )
            metrics[window_index, channel_index] = (
                peak_to_peak,
                rms,
                derivative_rms,
            )
            flatline[window_index, channel_index] = (
                peak_to_peak <= np.finfo(float).eps
            )

    valid_windows = ~skipped
    robust_z = np.full_like(metrics, np.nan)
    for channel_index in range(channel_count):
        for metric_index in range(len(metric_names)):
            robust_z[:, channel_index, metric_index] = _robust_upper_z(
                metrics[:, channel_index, metric_index],
                valid_windows,
            )

    candidates: list[ArtifactCandidate] = []
    for window_index in range(window_count):
        if skipped[window_index]:
            continue
        triggered = robust_z[window_index] > config.robust_z_threshold
        if not (
            np.any(triggered)
            or np.any(flatline[window_index])
            or np.any(nonfinite[window_index])
        ):
            continue
        trigger_details: list[str] = []
        high_channels: set[str] = set()
        derivative_channels: set[str] = set()
        flatline_channels: set[str] = set()
        nonfinite_channels: set[str] = set()
        for channel_index, channel_name in enumerate(config.channel_names):
            if nonfinite[window_index, channel_index]:
                nonfinite_channels.add(channel_name)
                trigger_details.append(f"{channel_name}包含NaN或Inf")
            if flatline[window_index, channel_index]:
                flatline_channels.add(channel_name)
                trigger_details.append(f"{channel_name}为完全平直线")
            for metric_index, metric_name in enumerate(metric_names):
                if triggered[channel_index, metric_index]:
                    high_channels.add(channel_name)
                    if metric_name == "derivative_rms":
                        derivative_channels.add(channel_name)
                    score = robust_z[window_index, channel_index, metric_index]
                    score_text = "inf" if np.isinf(score) else f"{score:.3f}"
                    trigger_details.append(
                        f"{channel_name} {metric_name}稳健异常分数={score_text}"
                    )
        possible_types, reason = _candidate_types_and_reason(
            config.channel_names,
            trigger_details,
            high_channels,
            derivative_channels,
            flatline_channels,
            nonfinite_channels,
        )
        start_sample = window_index * samples_per_window
        stop_sample = min(start_sample + samples_per_window, raw.n_times)
        candidates.append(
            ArtifactCandidate(
                candidate_id=f"CANDIDATE_{len(candidates) + 1:04d}",
                onset_seconds=start_sample / sfreq,
                duration_seconds=(stop_sample - start_sample) / sfreq,
                possible_types=possible_types,
                triggered_metrics=tuple(trigger_details),
                algorithm_reason=reason,
            )
        )
    return ArtifactDetectionResult(
        config=config,
        candidates=tuple(candidates),
        window_count=window_count,
        skipped_existing_bad_count=int(np.sum(skipped)),
    )


def attach_candidate_annotations(
    raw: mne.io.BaseRaw,
    detection: ArtifactDetectionResult,
) -> mne.io.BaseRaw:
    """Return a copy with non-BAD candidate annotations only."""

    result = raw.copy()
    if not detection.candidates:
        return result
    existing = set(str(value) for value in result.annotations.description)
    duplicate = [
        item.candidate_id for item in detection.candidates if item.candidate_id in existing
    ]
    if duplicate:
        raise ValueError(f"候选Annotation已存在：{', '.join(duplicate)}")
    candidate_annotations = mne.Annotations(
        onset=[item.onset_seconds for item in detection.candidates],
        duration=[item.duration_seconds for item in detection.candidates],
        description=[item.candidate_id for item in detection.candidates],
        orig_time=result.annotations.orig_time,
    )
    result.set_annotations(result.annotations + candidate_annotations)
    return result


@lru_cache(maxsize=1)
def _chinese_font() -> font_manager.FontProperties:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        if name in available:
            return font_manager.FontProperties(family=name)
    raise RuntimeError(
        "没有检测到可用中文字体。请用户手动安装中文字体后重试；"
        "Skill不会静默下载或安装字体。"
    )


def _plot_candidate(
    raw: mne.io.BaseRaw,
    candidate: ArtifactCandidate,
    config: ArtifactDetectionConfig,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    sfreq = float(raw.info["sfreq"])
    total_duration = raw.n_times / sfreq
    display_start = max(0.0, candidate.onset_seconds - config.context_seconds)
    display_stop = min(
        total_duration,
        candidate.onset_seconds
        + candidate.duration_seconds
        + config.context_seconds,
    )
    start_sample = max(0, int(math.floor(display_start * sfreq)))
    stop_sample = min(raw.n_times, int(math.ceil(display_stop * sfreq)))
    data = raw.get_data(
        picks=list(config.channel_names),
        start=start_sample,
        stop=stop_sample,
        reject_by_annotation=None,
    ) * 1e6
    times = np.arange(start_sample, stop_sample, dtype=float) / sfreq
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(config.channel_names),
        1,
        figsize=(13, 3.1 * len(config.channel_names)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    chinese_font = _chinese_font()
    for channel_index, channel_name in enumerate(config.channel_names):
        axis = axes[channel_index, 0]
        axis.plot(times, data[channel_index], color="#1F4E79", linewidth=0.75)
        axis.axvspan(
            candidate.onset_seconds,
            candidate.onset_seconds + candidate.duration_seconds,
            color="#D73027",
            alpha=0.18,
        )
        axis.set_ylabel("幅度（微伏）", fontproperties=chinese_font)
        axis.set_title(f"{channel_name}波形", fontproperties=chinese_font)
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axes[-1, 0].set_xlabel("合并后时间（秒）", fontproperties=chinese_font)
    figure.suptitle(
        f"候选伪迹 {candidate.candidate_id[-4:]}："
        f"{candidate.onset_seconds:g}至"
        f"{candidate.onset_seconds + candidate.duration_seconds:g}秒",
        fontproperties=chinese_font,
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def write_artifact_review_bundle(
    raw: mne.io.BaseRaw,
    detection: ArtifactDetectionResult,
    directory: Path,
    *,
    now: datetime,
) -> ArtifactReviewBundle:
    """Write method guidance, candidate data, images, and review placeholders."""

    directory = directory.resolve()
    method_path = next_versioned_path(
        directory,
        "伪迹检测方法与人工判断说明",
        ".md",
        now,
    )
    candidates_csv_path = next_versioned_path(
        directory,
        "候选伪迹指标与理由",
        ".csv",
        now,
    )
    review_manifest_path = next_versioned_path(
        directory,
        "候选伪迹逐图审核清单",
        ".json",
        now,
    )
    review_markdown_path = next_versioned_path(
        directory,
        "候选伪迹逐图审核报告",
        ".md",
        now,
    )
    write_bytes_exclusive(
        method_path,
        artifact_method_explanation(detection.config).encode("utf-8"),
    )

    image_paths = []
    manifest_candidates = []
    rows = []
    for candidate in detection.candidates:
        image_path = next_versioned_path(
            directory,
            f"候选伪迹{candidate.candidate_id[-4:]}波形",
            ".png",
            now,
        )
        _plot_candidate(raw, candidate, detection.config, image_path)
        image_paths.append(image_path)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "onset_seconds": candidate.onset_seconds,
                "duration_seconds": candidate.duration_seconds,
                "possible_types": "；".join(candidate.possible_types),
                "triggered_metrics": "；".join(candidate.triggered_metrics),
                "algorithm_reason": candidate.algorithm_reason,
                "image_path": str(image_path),
            }
        )
        manifest_candidates.append(
            {
                **asdict(candidate),
                "image_path": str(image_path),
                "codex_review": {
                    "status": "pending",
                    "verdict": "",
                    "artifact_type": "",
                    "reason": "",
                    "image_evidence": "",
                },
            }
        )

    fields = [
        "candidate_id",
        "onset_seconds",
        "duration_seconds",
        "possible_types",
        "triggered_metrics",
        "algorithm_reason",
        "image_path",
    ]
    write_bytes_exclusive(candidates_csv_path, _csv_bytes(fields, rows))
    manifest = {
        "method": asdict(detection.config),
        "window_count": detection.window_count,
        "skipped_existing_bad_count": detection.skipped_existing_bad_count,
        "candidate_count": len(detection.candidates),
        "candidate_duration_seconds": sum(
            item.duration_seconds for item in detection.candidates
        ),
        "candidates": manifest_candidates,
    }
    write_bytes_exclusive(
        review_manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )

    lines = [
        "# 候选伪迹逐图审核报告",
        "",
        f"- 候选数量：{len(detection.candidates)}",
        f"- 候选总时长：{manifest['candidate_duration_seconds']:g}秒",
        "- 红色阴影表示算法选中的5秒窗口；阴影外为判断上下文。",
        "- 算法理由不是最终伪迹诊断，必须完成Codex逐图复核后再询问用户。",
        "",
    ]
    if not detection.candidates:
        lines.extend(["本次未检出候选伪迹。", ""])
    for candidate, image_path in zip(detection.candidates, image_paths):
        lines.extend(
            [
                f"## {candidate.candidate_id}",
                "",
                f"![{candidate.candidate_id}波形]({image_path.name})",
                "",
                f"- 算法入选理由：{candidate.algorithm_reason}",
                f"- 可能伪迹类型：{'；'.join(candidate.possible_types)}",
                "- Codex逐图复核结论：待填写",
                "- 判断伪迹的理由：待填写",
                "- 图像证据：待填写",
                "",
            ]
        )
    write_bytes_exclusive(
        review_markdown_path,
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    return ArtifactReviewBundle(
        method_path=method_path,
        candidates_csv_path=candidates_csv_path,
        review_manifest_path=review_manifest_path,
        review_markdown_path=review_markdown_path,
        image_paths=tuple(image_paths),
    )


def validate_completed_reviews(
    detection: ArtifactDetectionResult,
    reviews: tuple[ArtifactReview, ...],
) -> list[str]:
    errors: list[str] = []
    expected = {item.candidate_id for item in detection.candidates}
    seen: set[str] = set()
    valid_verdicts = {"支持伪迹", "不支持伪迹", "无法判断"}
    for review in reviews:
        if review.candidate_id in seen:
            errors.append(f"逐图复核ID重复：{review.candidate_id}")
        seen.add(review.candidate_id)
        if review.candidate_id not in expected:
            errors.append(f"逐图复核包含未知候选：{review.candidate_id}")
        if review.verdict not in valid_verdicts:
            errors.append(f"逐图复核结论无效：{review.candidate_id}")
        if not review.artifact_type.strip():
            errors.append(f"伪迹类型不能为空：{review.candidate_id}")
        if not review.reason.strip():
            errors.append(f"判断理由不能为空：{review.candidate_id}")
        if not review.image_evidence.strip():
            errors.append(f"图像证据不能为空：{review.candidate_id}")
    for candidate_id in sorted(expected.difference(seen)):
        errors.append(f"缺少逐图复核：{candidate_id}")
    return list(dict.fromkeys(errors))


def apply_artifact_batch_decision(
    raw: mne.io.BaseRaw,
    detection: ArtifactDetectionResult,
    reviews: tuple[ArtifactReview, ...],
    *,
    decision: ArtifactBatchDecision,
) -> ArtifactDecisionResult:
    """Apply one user decision without deleting or modifying signal samples."""

    if decision not in {"exclude_supported", "retain_all"}:
        raise ValueError(f"未知伪迹批量决策：{decision}")
    review_errors = validate_completed_reviews(detection, reviews)
    if review_errors:
        raise ValueError("逐图复核未完成：" + "；".join(review_errors))
    result = raw.copy()
    supported_ids = {
        review.candidate_id
        for review in reviews
        if review.verdict == "支持伪迹"
    }
    selected = (
        [
            item
            for item in detection.candidates
            if item.candidate_id in supported_ids
        ]
        if decision == "exclude_supported"
        else []
    )
    if selected:
        existing = set(str(value) for value in result.annotations.description)
        descriptions = [
            f"BAD_auto_{item.candidate_id[-4:]}" for item in selected
        ]
        duplicate = [value for value in descriptions if value in existing]
        if duplicate:
            raise ValueError(f"自动BAD Annotation已存在：{', '.join(duplicate)}")
        bad_annotations = mne.Annotations(
            onset=[item.onset_seconds for item in selected],
            duration=[item.duration_seconds for item in selected],
            description=descriptions,
            orig_time=result.annotations.orig_time,
        )
        result.set_annotations(result.annotations + bad_annotations)
    return ArtifactDecisionResult(
        raw=result,
        decision=decision,
        bad_count=len(selected),
        bad_duration_seconds=float(sum(item.duration_seconds for item in selected)),
    )


def write_completed_artifact_reviews(
    detection: ArtifactDetectionResult,
    reviews: tuple[ArtifactReview, ...],
    image_paths: tuple[Path, ...],
    directory: Path,
    *,
    now: datetime,
) -> tuple[Path, Path]:
    """Freeze Codex's completed per-image reasoning without overwriting templates."""

    if len(image_paths) != len(detection.candidates):
        raise ValueError("候选数量与图片数量不一致")
    review_errors = validate_completed_reviews(detection, reviews)
    if review_errors:
        raise ValueError("逐图复核未完成：" + "；".join(review_errors))
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(missing_images[0])

    directory = directory.resolve()
    json_path = next_versioned_path(
        directory,
        "候选伪迹逐图审核完成版",
        ".json",
        now,
    )
    markdown_path = next_versioned_path(
        directory,
        "候选伪迹逐图审核完成版",
        ".md",
        now,
    )
    reviews_by_id = {review.candidate_id: review for review in reviews}
    payload_candidates = []
    lines = [
        "# 候选伪迹逐图审核完成版",
        "",
        f"- 候选数量：{len(detection.candidates)}",
        "- 每张图片均需由Codex查看，并与候选指标交叉核对。",
        "- 只有结论为“支持伪迹”的候选才能在用户确认后写入BAD_auto_。",
        "",
    ]
    for candidate, image_path in zip(detection.candidates, image_paths):
        review = reviews_by_id[candidate.candidate_id]
        payload_candidates.append(
            {
                **asdict(candidate),
                "image_path": str(image_path.resolve()),
                "codex_review": {
                    "status": "complete",
                    "verdict": review.verdict,
                    "artifact_type": review.artifact_type,
                    "reason": review.reason,
                    "image_evidence": review.image_evidence,
                },
            }
        )
        lines.extend(
            [
                f"## {candidate.candidate_id}",
                "",
                f"![{candidate.candidate_id}波形]({image_path.name})",
                "",
                f"- 算法入选理由：{candidate.algorithm_reason}",
                f"- 逐图复核结论：{review.verdict}",
                f"- 伪迹类型：{review.artifact_type}",
                f"- 判断伪迹的理由：{review.reason}",
                f"- 图像证据：{review.image_evidence}",
                "",
            ]
        )
    payload = {
        "method": asdict(detection.config),
        "candidate_count": len(detection.candidates),
        "candidates": payload_candidates,
    }
    write_bytes_exclusive(
        json_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    write_bytes_exclusive(
        markdown_path,
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    return json_path, markdown_path
