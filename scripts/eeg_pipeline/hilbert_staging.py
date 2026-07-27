"""Generate guarded EEG+EMG Hilbert candidate sleep-stage evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import io
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import hilbert

from .bandpower import BandpowerResult, BandpowerRow, PhaseInterval
from .line_noise import StepNotice, _chinese_font
from .workspace import next_versioned_path, write_bytes_exclusive


CANDIDATE_LABELS = (
    "Wake_candidate",
    "NREM_candidate",
    "REM_candidate",
    "Uncertain",
    "Artifact",
    "Boundary_Unscored",
)


@dataclass(frozen=True)
class HilbertConfig:
    eeg_channel_name: str = "EEG_bipolar"
    emg_channel_name: str = "EMG_bipolar"
    median_filter_samples: int = 40
    padding_seconds: float = 10.0
    exploratory_method_accepted: bool = False
    smoke_test_accepted: bool = False
    manual_review_bouts_per_candidate_stage: int = 10
    manual_review_context_seconds: float = 30.0


@dataclass(frozen=True)
class SmokeWindow:
    name: str
    start_seconds: float
    stop_seconds: float


@dataclass(frozen=True)
class HilbertFeatureRow:
    start_seconds: float
    stop_seconds: float
    median_uv: float
    mean_uv: float
    p95_uv: float
    p99_uv: float


@dataclass(frozen=True)
class CandidateThresholds:
    calibration_name: str
    calibration_start_seconds: float
    calibration_stop_seconds: float
    valid_window_count: int
    delta_q25: float
    delta_q50: float
    delta_q75: float
    theta_delta_q25: float
    theta_delta_q50: float
    theta_delta_q75: float
    sigma_q25: float
    sigma_q50: float
    sigma_q75: float
    high_gamma_q25: float
    high_gamma_q50: float
    high_gamma_q75: float
    hilbert_q25: float
    hilbert_q50: float
    hilbert_q75: float


@dataclass(frozen=True)
class HilbertCandidateRow:
    start_seconds: float
    stop_seconds: float
    midpoint_seconds: float
    phase: str
    source_status: str
    delta_absolute_v2: float
    theta_absolute_v2: float
    sigma_absolute_v2: float
    high_gamma_absolute_v2: float
    theta_delta_ratio: float
    hilbert_median_uv: float
    hilbert_mean_uv: float
    hilbert_p95_uv: float
    hilbert_p99_uv: float
    hilbert_calibration_band: str
    wake_rule_matched: bool
    nrem_rule_matched: bool
    rem_rule_matched: bool
    strict_rule_match_count: int
    candidate_stage: str
    candidate_reason: str
    is_ground_truth: bool = False
    manual_review_status: str = "Not_reviewed"


@dataclass(frozen=True)
class CandidateBout:
    bout_index: int
    start_seconds: float
    stop_seconds: float
    duration_seconds: float
    phase: str
    candidate_stage: str
    source_window_count: int
    formal_sleep_duration: bool = False


@dataclass(frozen=True)
class HilbertStagingResult:
    config: HilbertConfig
    thresholds: CandidateThresholds
    features: tuple[HilbertFeatureRow, ...]
    rows: tuple[HilbertCandidateRow, ...]
    bouts: tuple[CandidateBout, ...]


@dataclass(frozen=True)
class HilbertEvidenceBundle:
    epoch_csv_path: Path
    bout_csv_path: Path
    phase_summary_csv_path: Path
    review_checklist_csv_path: Path
    config_path: Path
    report_path: Path


@dataclass(frozen=True)
class HilbertSmokeBundle:
    summary_csv_path: Path
    config_path: Path
    report_path: Path
    chart_paths: tuple[Path, ...]


def _validate_config(config: HilbertConfig) -> list[str]:
    errors = []
    if not config.eeg_channel_name.strip():
        errors.append("EEG通道不能为空")
    if not config.emg_channel_name.strip():
        errors.append("EMG通道不能为空")
    if config.eeg_channel_name == config.emg_channel_name:
        errors.append("EEG与EMG通道不能相同")
    if config.median_filter_samples <= 0:
        errors.append("Hilbert中值滤波采样点数必须大于0")
    if config.padding_seconds < 0:
        errors.append("Hilbert边缘填充秒数不能小于0")
    if config.manual_review_bouts_per_candidate_stage <= 0:
        errors.append("每类人工复核Bout数量必须大于0")
    if config.manual_review_context_seconds < 0:
        errors.append("人工复核上下文秒数不能小于0")
    return errors


def hilbert_step_notices(config: HilbertConfig) -> tuple[StepNotice, ...]:
    errors = _validate_config(config)
    if errors:
        raise ValueError("；".join(errors))
    return (
        StepNotice(1, 8, "确认Hilbert输入", "核对滤波冻结版、Bandpower、EMG通道、Marker、BAD和SHA256。", "保证信号与逐窗特征在同一时间轴。", "Hilbert输入QC。"),
        StepNotice(2, 8, "说明探索性边界", "先展示Hilbert方法、候选规则及限制，由用户确认接受。", "避免把候选标签误作正式睡眠分期。", "用户接受记录。"),
        StepNotice(3, 8, "运行Smoke Test", "只运行用户指定的短窗口，展示EEG、EMG包络与BAD。", "先发现通道、尺度或边缘问题。", "Smoke Test证据。"),
        StepNotice(4, 8, "确认Smoke Test", "用户查看短窗口后明确决定是否运行目标时段。", "防止烟雾测试尚未验收就直接全量运行。", "Smoke Test接受记录。"),
        StepNotice(5, 8, "提取Hilbert特征", "对EMG计算绝对Hilbert包络并作40点中值滤波。", "形成逐窗肌张力辅助特征。", "Hilbert逐窗特征。"),
        StepNotice(6, 8, "校准并生成候选", "在用户提供的校准区间计算Q25/Q50/Q75，再应用EEG主判定规则。", "生成可复核的Wake、NREM、REM候选。", "候选标签与规则证据。"),
        StepNotice(7, 8, "保留无效与冲突", "BAD写Artifact、边界写Boundary_Unscored、冲突或低置信写Uncertain。", "禁止强制归类和虚构概率。", "完整状态标签。"),
        StepNotice(8, 8, "保存Bout与证据", "按中点时间箱合并同阶段同标签连续Bout，不删除短Bout。", "支持人工复核与可追溯交付。", "逐窗、Bout、汇总、配置和报告。"),
    )


def hilbert_method_explanation(config: HilbertConfig) -> str:
    errors = _validate_config(config)
    if errors:
        raise ValueError("；".join(errors))
    return (
        "本阶段使用scipy.signal.hilbert计算EMG_bipolar解析信号的绝对值包络，"
        f"再进行{config.median_filter_samples}个采样点的中值滤波。Hilbert包络不会去除心电、"
        "噪声或肌电混杂，高包络只能作为活动或肌张力证据之一。固定的众数+2SD阈值"
        "没有通用稳定性判据，本版保持关闭；程序在用户提供的校准区间内，对有效窗口"
        "计算Q25/Q50/Q75，并以EEG Bandpower为主、EMG包络为辅生成"
        "Wake_candidate、NREM_candidate和REM_candidate。BAD必须输出Artifact，"
        "阶段边界输出Boundary_Unscored，冲突、缺失或低置信结果输出Uncertain。"
        "所有标签都不是Ground Truth，不能直接作为正式睡眠时长或N2O生物学结论。"
    )


def _validate_bandpower_geometry(result: BandpowerResult) -> None:
    if not result.rows:
        raise ValueError("Bandpower逐窗结果不能为空")
    expected = (
        ("窗口", result.window_seconds, 5.0),
        ("步长", result.step_seconds, 2.5),
        ("Welch子窗", result.welch_seconds, 2.0),
    )
    mismatched = [name for name, actual, wanted in expected if not np.isclose(actual, wanted)]
    if mismatched:
        raise ValueError("Hilbert候选规则只允许已批准的5秒窗、2.5秒步长和2秒Welch配置")


def _validate_row_alignment(
    bandpower: BandpowerResult,
    features: tuple[HilbertFeatureRow, ...],
) -> None:
    if len(features) != len(bandpower.rows):
        raise ValueError("Hilbert特征行数必须与Bandpower行数一致")
    for power_row, feature_row in zip(bandpower.rows, features):
        if not (
            np.isclose(power_row.start_seconds, feature_row.start_seconds)
            and np.isclose(power_row.stop_seconds, feature_row.stop_seconds)
        ):
            raise ValueError("Hilbert特征与Bandpower窗口时间不一致")


def extract_hilbert_features(
    raw: mne.io.BaseRaw,
    bandpower: BandpowerResult,
    config: HilbertConfig,
) -> tuple[HilbertFeatureRow, ...]:
    errors = _validate_config(config)
    if config.emg_channel_name not in raw.ch_names:
        errors.append(f"缺少EMG通道：{config.emg_channel_name}")
    if errors:
        raise ValueError("；".join(errors))
    _validate_bandpower_geometry(bandpower)
    sfreq = float(raw.info["sfreq"])
    first_start = min(row.start_seconds for row in bandpower.rows)
    last_stop = max(row.stop_seconds for row in bandpower.rows)
    if first_start < 0 or last_stop > raw.n_times / sfreq:
        raise ValueError("Bandpower窗口超出FIF时间范围")
    padding_samples = int(round(config.padding_seconds * sfreq))
    span_start = max(0, int(np.floor(first_start * sfreq)) - padding_samples)
    span_stop = min(raw.n_times, int(np.ceil(last_stop * sfreq)) + padding_samples)
    emg_uv = raw.get_data(
        picks=[config.emg_channel_name],
        start=span_start,
        stop=span_stop,
        reject_by_annotation=None,
    )[0] * 1e6
    envelope_uv = np.abs(hilbert(emg_uv))
    smoothed_uv = median_filter(
        envelope_uv,
        size=config.median_filter_samples,
        mode="nearest",
    )
    features = []
    for row in bandpower.rows:
        local_start = int(round(row.start_seconds * sfreq)) - span_start
        local_stop = int(round(row.stop_seconds * sfreq)) - span_start
        values = smoothed_uv[local_start:local_stop]
        if values.size == 0:
            raise ValueError("Hilbert窗口没有可用采样点")
        features.append(
            HilbertFeatureRow(
                start_seconds=row.start_seconds,
                stop_seconds=row.stop_seconds,
                median_uv=float(np.nanmedian(values)),
                mean_uv=float(np.nanmean(values)),
                p95_uv=float(np.nanpercentile(values, 95)),
                p99_uv=float(np.nanpercentile(values, 99)),
            )
        )
    return tuple(features)


def _bad_intervals(raw: mne.io.BaseRaw) -> tuple[tuple[float, float], ...]:
    intervals = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if str(description).upper().startswith("BAD") and duration > 0:
            start = float(onset) - float(raw.first_time)
            intervals.append((start, start + float(duration)))
    return tuple(intervals)


def _smoke_window_arrays(
    raw: mne.io.BaseRaw,
    config: HilbertConfig,
    window: SmokeWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    padding = int(round(config.padding_seconds * sfreq))
    start_sample = int(round(window.start_seconds * sfreq))
    stop_sample = int(round(window.stop_seconds * sfreq))
    padded_start = max(0, start_sample - padding)
    padded_stop = min(raw.n_times, stop_sample + padding)
    emg_padded_uv = raw.get_data(
        picks=[config.emg_channel_name],
        start=padded_start,
        stop=padded_stop,
        reject_by_annotation=None,
    )[0] * 1e6
    envelope = median_filter(
        np.abs(hilbert(emg_padded_uv)),
        size=config.median_filter_samples,
        mode="nearest",
    )
    local_start = start_sample - padded_start
    local_stop = stop_sample - padded_start
    eeg_uv = raw.get_data(
        picks=[config.eeg_channel_name],
        start=start_sample,
        stop=stop_sample,
        reject_by_annotation=None,
    )[0] * 1e6
    emg_uv = emg_padded_uv[local_start:local_stop]
    envelope_uv = envelope[local_start:local_stop]
    times = np.arange(eeg_uv.size, dtype=float) / sfreq + window.start_seconds
    return times, eeg_uv, emg_uv, envelope_uv


def _plot_smoke_window(
    raw: mne.io.BaseRaw,
    window: SmokeWindow,
    times: np.ndarray,
    eeg_uv: np.ndarray,
    emg_uv: np.ndarray,
    envelope_uv: np.ndarray,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = _chinese_font()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(times, eeg_uv, color="#1F4E79", linewidth=0.75)
    axes[1].plot(times, emg_uv, color="#8B3A3A", linewidth=0.65)
    axes[2].plot(times, envelope_uv, color="#237A57", linewidth=1.0)
    labels = ("EEG（µV）", "EMG（µV）", "Hilbert包络（µV）")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label, fontproperties=font)
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
        for bad_start, bad_stop in _bad_intervals(raw):
            left = max(window.start_seconds, bad_start)
            right = min(window.stop_seconds, bad_stop)
            if left < right:
                axis.axvspan(left, right, color="#C62828", alpha=0.15)
    axes[-1].set_xlabel("合并时间轴（秒）", fontproperties=font)
    figure.suptitle(
        f"Hilbert窗口检查-{window.name}",
        fontproperties=font,
        fontsize=14,
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def write_hilbert_smoke_evidence(
    raw: mne.io.BaseRaw,
    config: HilbertConfig,
    windows: tuple[SmokeWindow, ...],
    directory: Path,
    *,
    now: datetime,
) -> HilbertSmokeBundle:
    errors = _validate_config(config)
    for channel in (config.eeg_channel_name, config.emg_channel_name):
        if channel not in raw.ch_names:
            errors.append(f"缺少Smoke Test通道：{channel}")
    if not config.exploratory_method_accepted:
        errors.append("用户尚未接受Hilbert探索性方法边界")
    if not windows:
        errors.append("Smoke Test窗口必须由用户提供")
    duration = raw.n_times / float(raw.info["sfreq"])
    names = []
    for window in windows:
        names.append(window.name.strip())
        if not window.name.strip():
            errors.append("Smoke Test窗口名称不能为空")
        if not (0 <= window.start_seconds < window.stop_seconds <= duration):
            errors.append(f"Smoke Test窗口越界：{window.name}")
    if len(names) != len(set(names)):
        errors.append("Smoke Test窗口名称不能重复")
    if errors:
        raise ValueError("；".join(dict.fromkeys(errors)))
    directory = directory.resolve()
    summary_path = next_versioned_path(directory, "Hilbert窗口检查汇总", ".csv", now)
    config_path = next_versioned_path(directory, "Hilbert窗口检查配置", ".json", now)
    report_path = next_versioned_path(directory, "Hilbert窗口检查报告", ".md", now)
    rows = []
    charts = []
    for window in windows:
        times, eeg_uv, emg_uv, envelope_uv = _smoke_window_arrays(raw, config, window)
        chart_path = next_versioned_path(
            directory,
            f"Hilbert窗口检查-{window.name}",
            ".png",
            now,
        )
        _plot_smoke_window(
            raw,
            window,
            times,
            eeg_uv,
            emg_uv,
            envelope_uv,
            chart_path,
        )
        charts.append(chart_path)
        rows.append(
            {
                "window_name": window.name,
                "start_seconds": window.start_seconds,
                "stop_seconds": window.stop_seconds,
                "hilbert_median_uv": float(np.nanmedian(envelope_uv)),
                "hilbert_mean_uv": float(np.nanmean(envelope_uv)),
                "hilbert_p95_uv": float(np.nanpercentile(envelope_uv, 95)),
                "bad_overlap": any(
                    window.start_seconds < stop and window.stop_seconds > start
                    for start, stop in _bad_intervals(raw)
                ),
                "chart_path": str(chart_path),
            }
        )
    write_bytes_exclusive(summary_path, _csv_bytes(list(rows[0]), rows))
    payload = {
        "full_run_allowed": False,
        "user_smoke_confirmation": None,
        "window_source": "user_supplied",
        "eeg_channel": config.eeg_channel_name,
        "emg_channel": config.emg_channel_name,
        "hilbert_function": "scipy.signal.hilbert",
        "median_filter_samples": config.median_filter_samples,
        "padding_seconds": config.padding_seconds,
        "charts": [str(path) for path in charts],
    }
    write_bytes_exclusive(
        config_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    report = (
        "# Hilbert窗口检查报告\n\n"
        "- 每张图依次显示EEG、EMG和Hilbert包络；红色背景表示BAD重叠。\n"
        "- Hilbert包络不会去除心电、噪声或肌电混杂。\n"
        "- 请检查通道是否合理、包络是否被极端尖峰支配、BAD是否正确显示。\n"
        "- 查看完成后请关闭图片后明确确认是否允许运行目标时段。\n"
        "- 未确认前，配置中的full_run_allowed保持false。\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    return HilbertSmokeBundle(
        summary_csv_path=summary_path,
        config_path=config_path,
        report_path=report_path,
        chart_paths=tuple(charts),
    )


def _required_power(row: BandpowerRow, name: str) -> float:
    if name not in row.absolute_power:
        return float("nan")
    return float(row.absolute_power[name])


def calibrate_candidate_thresholds(
    bandpower: BandpowerResult,
    features: tuple[HilbertFeatureRow, ...],
    calibration: PhaseInterval,
) -> CandidateThresholds:
    _validate_row_alignment(bandpower, features)
    if not calibration.name.strip():
        raise ValueError("校准区间名称不能为空")
    if calibration.start_seconds < 0 or calibration.stop_seconds <= calibration.start_seconds:
        raise ValueError("校准区间无效")
    values = {
        "delta": [],
        "theta_delta": [],
        "sigma": [],
        "high_gamma": [],
        "hilbert": [],
    }
    for row, feature in zip(bandpower.rows, features):
        if row.status != "Valid":
            continue
        if not (
            row.start_seconds >= calibration.start_seconds
            and row.stop_seconds <= calibration.stop_seconds
        ):
            continue
        delta = _required_power(row, "Delta")
        theta = _required_power(row, "Theta")
        sigma = _required_power(row, "Sigma")
        high_gamma = _required_power(row, "High_Gamma")
        candidate = (delta, theta, sigma, high_gamma, feature.median_uv)
        if not all(np.isfinite(value) for value in candidate) or delta <= 0:
            continue
        values["delta"].append(delta)
        values["theta_delta"].append(theta / delta)
        values["sigma"].append(sigma)
        values["high_gamma"].append(high_gamma)
        values["hilbert"].append(feature.median_uv)
    count = len(values["delta"])
    if count == 0:
        raise ValueError("用户提供的校准区间没有有效窗口")

    def quartiles(name: str) -> tuple[float, float, float]:
        result = np.quantile(np.asarray(values[name], dtype=float), (0.25, 0.5, 0.75))
        return tuple(float(value) for value in result)

    delta = quartiles("delta")
    theta_delta = quartiles("theta_delta")
    sigma = quartiles("sigma")
    high_gamma = quartiles("high_gamma")
    hilbert_values = quartiles("hilbert")
    return CandidateThresholds(
        calibration_name=calibration.name,
        calibration_start_seconds=calibration.start_seconds,
        calibration_stop_seconds=calibration.stop_seconds,
        valid_window_count=count,
        delta_q25=delta[0],
        delta_q50=delta[1],
        delta_q75=delta[2],
        theta_delta_q25=theta_delta[0],
        theta_delta_q50=theta_delta[1],
        theta_delta_q75=theta_delta[2],
        sigma_q25=sigma[0],
        sigma_q50=sigma[1],
        sigma_q75=sigma[2],
        high_gamma_q25=high_gamma[0],
        high_gamma_q50=high_gamma[1],
        high_gamma_q75=high_gamma[2],
        hilbert_q25=hilbert_values[0],
        hilbert_q50=hilbert_values[1],
        hilbert_q75=hilbert_values[2],
    )


def _hilbert_band(value: float, thresholds: CandidateThresholds) -> str:
    if not np.isfinite(value):
        return "Missing"
    if value <= thresholds.hilbert_q25:
        return "at_or_below_Q25"
    if value <= thresholds.hilbert_q50:
        return "Q25_to_Q50"
    if value < thresholds.hilbert_q75:
        return "Q50_to_Q75"
    return "at_or_above_Q75"


def _forced_candidate(
    row: BandpowerRow,
    feature: HilbertFeatureRow,
    stage: str,
    reason: str,
) -> HilbertCandidateRow:
    delta = _required_power(row, "Delta")
    theta = _required_power(row, "Theta")
    return HilbertCandidateRow(
        start_seconds=row.start_seconds,
        stop_seconds=row.stop_seconds,
        midpoint_seconds=row.midpoint_seconds,
        phase=row.phase,
        source_status=row.status,
        delta_absolute_v2=delta,
        theta_absolute_v2=theta,
        sigma_absolute_v2=_required_power(row, "Sigma"),
        high_gamma_absolute_v2=_required_power(row, "High_Gamma"),
        theta_delta_ratio=theta / delta if np.isfinite(delta) and delta > 0 else float("nan"),
        hilbert_median_uv=feature.median_uv,
        hilbert_mean_uv=feature.mean_uv,
        hilbert_p95_uv=feature.p95_uv,
        hilbert_p99_uv=feature.p99_uv,
        hilbert_calibration_band="Not_applicable",
        wake_rule_matched=False,
        nrem_rule_matched=False,
        rem_rule_matched=False,
        strict_rule_match_count=0,
        candidate_stage=stage,
        candidate_reason=reason,
    )


def classify_candidate_rows(
    bandpower: BandpowerResult,
    features: tuple[HilbertFeatureRow, ...],
    thresholds: CandidateThresholds,
) -> tuple[HilbertCandidateRow, ...]:
    _validate_row_alignment(bandpower, features)
    output = []
    for row, feature in zip(bandpower.rows, features):
        if row.status == "Artifact":
            output.append(_forced_candidate(row, feature, "Artifact", "source_artifact_status"))
            continue
        if row.status == "Boundary_Unscored":
            output.append(_forced_candidate(row, feature, "Boundary_Unscored", "source_boundary_status"))
            continue
        delta = _required_power(row, "Delta")
        theta = _required_power(row, "Theta")
        sigma = _required_power(row, "Sigma")
        high_gamma = _required_power(row, "High_Gamma")
        required = (delta, theta, sigma, high_gamma, feature.median_uv)
        if not all(np.isfinite(value) for value in required) or delta <= 0:
            output.append(_forced_candidate(row, feature, "Uncertain", "missing_or_invalid_feature"))
            continue
        theta_delta = theta / delta
        wake = (
            feature.median_uv >= thresholds.hilbert_q75
            and delta <= thresholds.delta_q50
        ) or (
            high_gamma >= thresholds.high_gamma_q75
            and delta <= thresholds.delta_q25
            and feature.median_uv >= thresholds.hilbert_q50
        )
        nrem = (
            delta >= thresholds.delta_q75
            or (
                sigma >= thresholds.sigma_q75
                and delta >= thresholds.delta_q50
            )
        ) and (
            theta_delta <= thresholds.theta_delta_q50
            and feature.median_uv < thresholds.hilbert_q75
        )
        rem = (
            theta_delta >= thresholds.theta_delta_q75
            and delta <= thresholds.delta_q25
            and feature.median_uv <= thresholds.hilbert_q25
        )
        matches = int(wake) + int(nrem) + int(rem)
        if matches == 1:
            stage = (
                "Wake_candidate"
                if wake
                else "NREM_candidate"
                if nrem
                else "REM_candidate"
            )
            reason = "single_strict_rule_match"
        elif matches > 1:
            stage = "Uncertain"
            reason = "conflicting_strict_rules"
        else:
            stage = "Uncertain"
            reason = "no_strict_rule_match"
        output.append(
            HilbertCandidateRow(
                start_seconds=row.start_seconds,
                stop_seconds=row.stop_seconds,
                midpoint_seconds=row.midpoint_seconds,
                phase=row.phase,
                source_status=row.status,
                delta_absolute_v2=delta,
                theta_absolute_v2=theta,
                sigma_absolute_v2=sigma,
                high_gamma_absolute_v2=high_gamma,
                theta_delta_ratio=theta_delta,
                hilbert_median_uv=feature.median_uv,
                hilbert_mean_uv=feature.mean_uv,
                hilbert_p95_uv=feature.p95_uv,
                hilbert_p99_uv=feature.p99_uv,
                hilbert_calibration_band=_hilbert_band(feature.median_uv, thresholds),
                wake_rule_matched=bool(wake),
                nrem_rule_matched=bool(nrem),
                rem_rule_matched=bool(rem),
                strict_rule_match_count=matches,
                candidate_stage=stage,
                candidate_reason=reason,
            )
        )
    return tuple(output)


def build_candidate_bouts(
    rows: tuple[HilbertCandidateRow, ...],
) -> tuple[CandidateBout, ...]:
    if not rows:
        return ()
    ordered = sorted(rows, key=lambda item: item.midpoint_seconds)
    boundaries = [ordered[0].start_seconds]
    boundaries.extend(
        (left.midpoint_seconds + right.midpoint_seconds) / 2
        for left, right in zip(ordered, ordered[1:])
    )
    boundaries.append(ordered[-1].stop_seconds)
    bins = [
        {
            "start": boundaries[index],
            "stop": boundaries[index + 1],
            "phase": row.phase,
            "stage": row.candidate_stage,
            "count": 1,
        }
        for index, row in enumerate(ordered)
    ]
    merged = []
    for item in bins:
        if (
            merged
            and merged[-1]["stage"] == item["stage"]
            and merged[-1]["phase"] == item["phase"]
            and np.isclose(merged[-1]["stop"], item["start"])
        ):
            merged[-1]["stop"] = item["stop"]
            merged[-1]["count"] += 1
        else:
            merged.append(dict(item))
    return tuple(
        CandidateBout(
            bout_index=index,
            start_seconds=float(item["start"]),
            stop_seconds=float(item["stop"]),
            duration_seconds=float(item["stop"] - item["start"]),
            phase=str(item["phase"]),
            candidate_stage=str(item["stage"]),
            source_window_count=int(item["count"]),
        )
        for index, item in enumerate(merged)
    )


def run_hilbert_candidate_staging(
    raw: mne.io.BaseRaw,
    bandpower: BandpowerResult,
    config: HilbertConfig,
    calibration: PhaseInterval,
) -> HilbertStagingResult:
    if not config.exploratory_method_accepted:
        raise ValueError("用户尚未接受Hilbert探索性方法边界")
    if not config.smoke_test_accepted:
        raise ValueError("Hilbert Smoke Test尚未由用户确认")
    features = extract_hilbert_features(raw, bandpower, config)
    thresholds = calibrate_candidate_thresholds(bandpower, features, calibration)
    rows = classify_candidate_rows(bandpower, features, thresholds)
    bouts = build_candidate_bouts(rows)
    return HilbertStagingResult(config, thresholds, features, rows, bouts)


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _phase_summary(result: HilbertStagingResult) -> list[dict]:
    phase_totals = {}
    grouped = {}
    for bout in result.bouts:
        phase_totals[bout.phase] = phase_totals.get(bout.phase, 0.0) + bout.duration_seconds
        key = (bout.phase, bout.candidate_stage)
        grouped[key] = grouped.get(key, 0.0) + bout.duration_seconds
    return [
        {
            "phase": phase,
            "candidate_stage": stage,
            "duration_seconds": duration,
            "percentage_of_phase": duration / phase_totals[phase] * 100,
            "formal_sleep_duration": False,
        }
        for (phase, stage), duration in sorted(grouped.items())
    ]



def _manual_review_rows(result: HilbertStagingResult) -> list[dict]:
    if not result.rows:
        return []
    analysis_start = min(row.start_seconds for row in result.rows)
    analysis_stop = max(row.stop_seconds for row in result.rows)
    selected = []
    for stage in ("Wake_candidate", "NREM_candidate", "REM_candidate"):
        candidates = sorted(
            (bout for bout in result.bouts if bout.candidate_stage == stage),
            key=lambda item: (-item.duration_seconds, item.start_seconds),
        )[: result.config.manual_review_bouts_per_candidate_stage]
        selected.extend(candidates)
    selected.sort(key=lambda item: (item.candidate_stage, -item.duration_seconds))
    return [
        {
            "bout_index": bout.bout_index,
            "candidate_stage": bout.candidate_stage,
            "phase": bout.phase,
            "bout_start_seconds": bout.start_seconds,
            "bout_stop_seconds": bout.stop_seconds,
            "bout_duration_seconds": bout.duration_seconds,
            "suggested_review_start_seconds": max(
                analysis_start,
                bout.start_seconds - result.config.manual_review_context_seconds,
            ),
            "suggested_review_stop_seconds": min(
                analysis_stop,
                bout.stop_seconds + result.config.manual_review_context_seconds,
            ),
            "review_decision": "",
            "reviewer": "",
            "review_notes": "",
            "review_status": "Pending",
        }
        for bout in selected
    ]

def write_hilbert_evidence(
    result: HilbertStagingResult,
    directory: Path,
    *,
    now: datetime,
) -> HilbertEvidenceBundle:
    directory = directory.resolve()
    epoch_path = next_versioned_path(directory, "Hilbert候选分期逐窗结果", ".csv", now)
    bout_path = next_versioned_path(directory, "Hilbert候选分期连续Bout", ".csv", now)
    summary_path = next_versioned_path(directory, "Hilbert候选分期阶段汇总", ".csv", now)
    config_path = next_versioned_path(directory, "Hilbert候选分期配置", ".json", now)
    report_path = next_versioned_path(directory, "Hilbert候选分期方法报告", ".md", now)
    epoch_rows = [asdict(row) for row in result.rows]
    write_bytes_exclusive(epoch_path, _csv_bytes(list(epoch_rows[0]), epoch_rows))
    bout_rows = [asdict(bout) for bout in result.bouts]
    write_bytes_exclusive(bout_path, _csv_bytes(list(bout_rows[0]), bout_rows))
    summary_rows = _phase_summary(result)
    write_bytes_exclusive(
        summary_path,
        _csv_bytes(list(summary_rows[0]), summary_rows),
    )
    review_path = next_versioned_path(directory, "Hilbert候选Bout人工复核清单", ".csv", now)
    review_fields = [
        "bout_index",
        "candidate_stage",
        "phase",
        "bout_start_seconds",
        "bout_stop_seconds",
        "bout_duration_seconds",
        "suggested_review_start_seconds",
        "suggested_review_stop_seconds",
        "review_decision",
        "reviewer",
        "review_notes",
        "review_status",
    ]
    write_bytes_exclusive(
        review_path,
        _csv_bytes(review_fields, _manual_review_rows(result)),
    )
    payload = {
        "formal_sleep_staging": False,
        "ground_truth": False,
        "fixed_wu_threshold_enabled": False,
        "fixed_wu_threshold_reason": "无通用稳定性判据，保持关闭",
        "emg_channel": result.config.emg_channel_name,
        "hilbert_function": "scipy.signal.hilbert",
        "envelope": "absolute analytic signal in microvolts",
        "median_filter_samples": result.config.median_filter_samples,
        "padding_seconds": result.config.padding_seconds,
        "calibration_quantiles": "Q25/Q50/Q75",
        "thresholds": asdict(result.thresholds),
        "labels": list(CANDIDATE_LABELS),
        "short_bout_deletion": False,
        "state_transition_model": False,
        "manual_review_bouts_per_candidate_stage": result.config.manual_review_bouts_per_candidate_stage,
        "manual_review_context_seconds": result.config.manual_review_context_seconds,
        "rules": {
            "REM_candidate": "theta_delta >= Q75 AND delta <= Q25 AND Hilbert <= Q25",
            "NREM_candidate": "(delta >= Q75 OR (sigma >= Q75 AND delta >= Q50)) AND theta_delta <= Q50 AND Hilbert < Q75",
            "Wake_candidate": "(Hilbert >= Q75 AND delta <= Q50) OR (high_gamma >= Q75 AND delta <= Q25 AND Hilbert >= Q50)",
            "Uncertain": "no strict rule, conflicting rules, or missing feature",
            "Artifact": "source Bandpower status is Artifact",
            "Boundary_Unscored": "source Bandpower status is Boundary_Unscored",
        },
    }
    write_bytes_exclusive(
        config_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    report = (
        "# Hilbert候选分期方法报告\n\n"
        "- 结果边界：本结果不是Ground Truth，也不是正式Wake/NREM/REM分期。\n"
        "- EMG方法：绝对Hilbert包络，40个采样点中值滤波；不会去除心电或噪声。\n"
        "- 校准方法：仅使用用户提供校准区间内的有效窗口计算Q25/Q50/Q75。\n"
        "- 判定方法：EEG Bandpower为主，Hilbert EMG包络为辅助。\n"
        "- 无效处理：Artifact与Boundary_Unscored保持原标签；冲突或低置信写Uncertain。\n"
        "- Bout方法：相邻窗口中心的中点形成非重叠时间箱；没有删除短Bout。\n"
        "- 复核清单：每类最多选择10个最长候选Bout，建议查看前后各30秒。\n"
        "- 人工审核：REM_candidate及长Bout必须交由有经验的同组人员复核。\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    return HilbertEvidenceBundle(
        epoch_csv_path=epoch_path,
        bout_csv_path=bout_path,
        phase_summary_csv_path=summary_path,
        review_checklist_csv_path=review_path,
        config_path=config_path,
        report_path=report_path,
    )
