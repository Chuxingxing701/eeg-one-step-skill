"""Compute paper-aligned mouse EEG Bandpower in reproducible windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import io
import json

import mne
import numpy as np
from scipy.signal import welch

from .line_noise import StepNotice
from .workspace import next_versioned_path, write_bytes_exclusive


@dataclass(frozen=True)
class BandDefinition:
    name: str
    fmin: float
    fmax: float


@dataclass(frozen=True)
class PhaseInterval:
    name: str
    start_seconds: float
    stop_seconds: float


@dataclass(frozen=True)
class BandpowerConfig:
    channel_name: str = "EEG_bipolar"
    window_seconds: float = 5.0
    step_seconds: float = 2.5
    welch_seconds: float = 2.0
    welch_overlap_fraction: float = 0.5
    total_fmin: float = 0.5
    total_fmax: float = 100.0


@dataclass(frozen=True)
class BandpowerRow:
    start_seconds: float
    stop_seconds: float
    midpoint_seconds: float
    phase: str
    status: str
    absolute_power: dict[str, float]
    relative_power: dict[str, float]


@dataclass(frozen=True)
class BandpowerResult:
    rows: tuple[BandpowerRow, ...]
    primary_bands: tuple[BandDefinition, ...]
    additional_bands: tuple[BandDefinition, ...]
    window_seconds: float
    step_seconds: float
    welch_seconds: float
    n_per_seg: int
    n_overlap: int
    n_fft: int


def _validate_config(config: BandpowerConfig) -> list[str]:
    errors = []
    if not config.channel_name.strip():
        errors.append("Bandpower通道不能为空")
    if config.window_seconds <= 0 or config.step_seconds <= 0:
        errors.append("Bandpower窗口和步长必须大于0")
    if config.welch_seconds <= 0 or config.welch_seconds > config.window_seconds:
        errors.append("Welch子窗必须大于0且不超过Bandpower窗口")
    if not 0 <= config.welch_overlap_fraction < 1:
        errors.append("Welch重叠比例必须大于等于0且小于1")
    if config.total_fmin < 0 or config.total_fmax <= config.total_fmin:
        errors.append("总功率频率范围无效")
    return errors


def bandpower_step_notices(config: BandpowerConfig) -> tuple[StepNotice, ...]:
    errors = _validate_config(config)
    if errors:
        raise ValueError("；".join(errors))
    return (
        StepNotice(1, 7, "确认Bandpower输入", "核对滤波冻结版、EEG通道、采样率、Marker与BAD。", "保证输入可追溯。", "输入QC。"),
        StepNotice(2, 7, "确认阶段区间", "只使用用户提供的阶段起止时间，不推断Baseline、Treatment或Recovery。", "为每个5秒窗分配实验阶段。", "PhaseInterval清单。"),
        StepNotice(3, 7, "声明频段与窗参数", "采用论文主频段、5秒窗、2.5秒步长及2秒Hann-Welch子窗。", "冻结计算口径。", "Bandpower配置。"),
        StepNotice(4, 7, "逐窗计算Bandpower", "每窗使用50%重叠的2秒Hann-Welch，nfft等于子窗样本数。", "得到绝对和相对频段功率。", "逐窗结果。"),
        StepNotice(5, 7, "处理无效窗口", "任何BAD重叠标Artifact；跨阶段标Boundary_Unscored；数值写NaN。", "避免把无效窗口误当成零功率。", "状态标签与NaN。"),
        StepNotice(6, 7, "保存证据", "输出逐窗CSV、参数JSON和方法报告，文件不可覆盖。", "支持复查和复现。", "Bandpower证据包。"),
        StepNotice(7, 7, "进入图表分析", "主频段与Alpha附加指标分开制图，并由Codex逐图解释。", "防止附加Alpha混入论文主结论。", "待分析图表输入。"),
    )


def paper_band_definitions() -> tuple[
    tuple[BandDefinition, ...],
    tuple[BandDefinition, ...],
]:
    primary = (
        BandDefinition("Delta", 0.5, 4.5),
        BandDefinition("Theta", 6.0, 9.0),
        BandDefinition("Sigma", 10.0, 15.0),
        BandDefinition("Beta", 15.0, 30.0),
        BandDefinition("Low_Gamma", 30.0, 50.0),
        BandDefinition("High_Gamma", 50.0, 100.0),
    )
    additional = (BandDefinition("Alpha_additional", 8.0, 12.0),)
    return primary, additional


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


def _overlaps(start: float, stop: float, intervals) -> bool:
    return any(start < right and stop > left for left, right in intervals)


def _validate_phases(phases: tuple[PhaseInterval, ...], duration: float) -> None:
    if not phases:
        raise ValueError("必须提供用户确认的实验阶段区间")
    ordered = sorted(phases, key=lambda item: item.start_seconds)
    for phase in ordered:
        if not phase.name.strip():
            raise ValueError("实验阶段名称不能为空")
        if not (0 <= phase.start_seconds < phase.stop_seconds <= duration):
            raise ValueError(f"实验阶段区间越界：{phase.name}")
    for left, right in zip(ordered, ordered[1:]):
        if right.start_seconds < left.stop_seconds:
            raise ValueError("实验阶段区间不得重叠")


def _phase_for_window(
    start: float,
    stop: float,
    phases: tuple[PhaseInterval, ...],
) -> tuple[str, str]:
    containing = [
        phase
        for phase in phases
        if start >= phase.start_seconds and stop <= phase.stop_seconds
    ]
    if len(containing) == 1:
        return containing[0].name, "Valid"
    return "Boundary", "Boundary_Unscored"


def _nan_powers(names: tuple[str, ...]) -> dict[str, float]:
    return {name: float("nan") for name in names}


def compute_bandpower(
    raw: mne.io.BaseRaw,
    config: BandpowerConfig,
    phases: tuple[PhaseInterval, ...],
) -> BandpowerResult:
    errors = _validate_config(config)
    if config.channel_name not in raw.ch_names:
        errors.append(f"缺少Bandpower通道：{config.channel_name}")
    sfreq = float(raw.info["sfreq"])
    if config.total_fmax > sfreq / 2:
        errors.append("Bandpower上限超过Nyquist频率")
    if errors:
        raise ValueError("；".join(errors))
    duration = raw.n_times / sfreq
    _validate_phases(phases, duration)
    window_samples = int(round(config.window_seconds * sfreq))
    step_samples = int(round(config.step_seconds * sfreq))
    n_per_seg = int(round(config.welch_seconds * sfreq))
    n_overlap = int(round(n_per_seg * config.welch_overlap_fraction))
    primary, additional = paper_band_definitions()
    all_bands = primary + additional
    names = tuple(item.name for item in all_bands)
    bad_intervals = _bad_intervals(raw)
    rows = []
    for start_sample in range(0, raw.n_times - window_samples + 1, step_samples):
        stop_sample = start_sample + window_samples
        start = start_sample / sfreq
        stop = stop_sample / sfreq
        phase, status = _phase_for_window(start, stop, phases)
        if _overlaps(start, stop, bad_intervals):
            status = "Artifact"
        if status != "Valid":
            rows.append(
                BandpowerRow(
                    start,
                    stop,
                    (start + stop) / 2,
                    phase,
                    status,
                    _nan_powers(names),
                    _nan_powers(names),
                )
            )
            continue
        data = raw.get_data(
            picks=[config.channel_name],
            start=start_sample,
            stop=stop_sample,
            reject_by_annotation=None,
        )[0]
        frequencies, psd = welch(
            data,
            fs=sfreq,
            window="hann",
            nperseg=n_per_seg,
            noverlap=n_overlap,
            nfft=n_per_seg,
            detrend="constant",
            scaling="density",
            average="mean",
        )
        total_mask = (
            (frequencies >= config.total_fmin)
            & (frequencies <= config.total_fmax)
        )
        total_power = float(np.trapezoid(psd[total_mask], frequencies[total_mask]))
        absolute = {}
        relative = {}
        for band in all_bands:
            mask = (frequencies >= band.fmin) & (frequencies <= band.fmax)
            power = float(np.trapezoid(psd[mask], frequencies[mask]))
            absolute[band.name] = power
            relative[band.name] = power / total_power if total_power > 0 else float("nan")
        rows.append(
            BandpowerRow(
                start,
                stop,
                (start + stop) / 2,
                phase,
                status,
                absolute,
                relative,
            )
        )
    return BandpowerResult(
        rows=tuple(rows),
        primary_bands=primary,
        additional_bands=additional,
        window_seconds=config.window_seconds,
        step_seconds=config.step_seconds,
        welch_seconds=config.welch_seconds,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        n_fft=n_per_seg,
    )


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def write_bandpower_evidence(
    result: BandpowerResult,
    directory: Path,
    *,
    now: datetime,
) -> tuple[Path, Path, Path]:
    directory = directory.resolve()
    csv_path = next_versioned_path(directory, "Bandpower逐窗结果", ".csv", now)
    config_path = next_versioned_path(directory, "Bandpower参数", ".json", now)
    report_path = next_versioned_path(directory, "Bandpower方法报告", ".md", now)
    band_names = tuple(
        item.name for item in result.primary_bands + result.additional_bands
    )
    fields = [
        "start_seconds",
        "stop_seconds",
        "midpoint_seconds",
        "phase",
        "status",
        *[f"{name}_absolute_V2" for name in band_names],
        *[f"{name}_relative" for name in band_names],
    ]
    rows = []
    for item in result.rows:
        row = {
            "start_seconds": item.start_seconds,
            "stop_seconds": item.stop_seconds,
            "midpoint_seconds": item.midpoint_seconds,
            "phase": item.phase,
            "status": item.status,
        }
        row.update(
            {f"{name}_absolute_V2": item.absolute_power[name] for name in band_names}
        )
        row.update(
            {f"{name}_relative": item.relative_power[name] for name in band_names}
        )
        rows.append(row)
    write_bytes_exclusive(csv_path, _csv_bytes(fields, rows))
    payload = {
        "window_seconds": result.window_seconds,
        "step_seconds": result.step_seconds,
        "welch_seconds": result.welch_seconds,
        "n_per_seg": result.n_per_seg,
        "n_overlap": result.n_overlap,
        "n_fft": result.n_fft,
        "welch_window": "hann",
        "invalid_value": "NaN",
        "primary_bands": [asdict(item) for item in result.primary_bands],
        "additional_bands": [asdict(item) for item in result.additional_bands],
    }
    write_bytes_exclusive(
        config_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    report = (
        "# Bandpower方法报告\n\n"
        "- 论文主频段：Delta、Theta、Sigma、Beta、Low Gamma、High Gamma。\n"
        "- Alpha仅为附加指标，不并入论文主频段结论。\n"
        "- 分析几何：5秒窗、2.5秒步长；窗内2秒Hann-Welch、50%重叠。\n"
        "- Artifact和Boundary_Unscored写为NaN，不写为0。\n"
        "- 阶段区间必须由用户提供，程序不根据名称或时长猜测。\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    return csv_path, config_path, report_path
