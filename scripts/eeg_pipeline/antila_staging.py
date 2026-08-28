"""Run the pinned Antila/PySleep staging method on a project window axis."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import pandas as pd

from .workspace import next_versioned_path, write_bytes_exclusive
from .line_noise import StepNotice


ANTILA_SOURCE_REPOSITORY = "https://github.com/tortugar/Lab"
ANTILA_SOURCE_COMMIT = "bcb8dae1594e64a511545e34f6050e2a417c1f45"
ANTILA_SLEEPY_SHA256 = "da7ad9b5a771362cc529cca98a19879837337d4e40969f684067bcacc0148a04"

AUTHOR_STATE_LABELS = np.asarray(("Uncertain", "REM", "Wake", "NREM"), dtype=object)
PROJECT_STATUSES = {"Valid", "Artifact", "Boundary_Unscored"}


@dataclass(frozen=True)
class AntilaSourceSpec:
    repository: str = ANTILA_SOURCE_REPOSITORY
    commit: str = ANTILA_SOURCE_COMMIT
    sleepy_sha256: str = ANTILA_SLEEPY_SHA256


@dataclass(frozen=True)
class AntilaConfig:
    source_root: Path
    data_root: Path
    recording: str
    calibration_mode: Literal["valid_only", "pooled_valid"] = "valid_only"
    paired_records_confirmed: bool = False
    pooled_calibration_approved: bool = False
    th_delta_std: float = 1.0
    mu_std: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.recording.strip():
            errors.append("Antila记录名称不能为空")
        if self.calibration_mode not in {"valid_only", "pooled_valid"}:
            errors.append("Antila校准模式无效")
        if self.calibration_mode == "pooled_valid" and not (
            self.paired_records_confirmed and self.pooled_calibration_approved
        ):
            errors.append("pooled Valid校准必须确认配对记录并取得用户批准")
        return errors


@dataclass(frozen=True)
class AntilaStagingResult:
    config: AntilaConfig
    windows: pd.DataFrame
    author_tail_windows: pd.DataFrame
    source_path: Path
    source_details: tuple[str, ...]


@dataclass(frozen=True)
class AntilaEvidenceBundle:
    window_csv_path: Path
    tail_csv_path: Path
    summary_csv_path: Path
    config_json_path: Path
    method_report_path: Path


@dataclass(frozen=True)
class AntilaFeatureArrays:
    delta: np.ndarray
    theta: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    emg: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(np.asarray(values)) for values in self.as_tuple()}
        if len(lengths) != 1:
            raise ValueError("Antila特征数组长度必须一致")

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        return (self.delta, self.theta, self.sigma, self.gamma, self.emg)


@dataclass(frozen=True)
class AntilaThresholds:
    theta_delta: float
    emg: float
    delta: float
    sigma: float
    gamma: float


def antila_step_notices() -> tuple[StepNotice, ...]:
    return (
        StepNotice(1, 8, "确认Antila输入", "核对滤波FIF、频段功率窗口轴、双极通道、Marker、status和SHA256。", "保证信号与项目窗口严格同轴。", "Antila输入QC。"),
        StepNotice(2, 8, "确认作者源码", "核对tortugar/Lab本地checkout、commit和sleepy.py SHA256。", "固定作者方法来源。", "Antila源码证据。"),
        StepNotice(3, 8, "确认校准范围", "默认使用当前记录Valid窗；配对pooled模式需用户批准。", "避免无效窗和未确认跨记录阈值进入校准。", "校准决定。"),
        StepNotice(4, 8, "准备作者输入", "版本化输出EEG.mat、EMG.mat、info.txt和频谱。", "形成作者程序所需输入。", "MAT与频谱证据。"),
        StepNotice(5, 8, "运行作者方法", "调用固定版本calculate_spectrum()和sleep_state()。", "得到作者窗口状态。", "作者运行记录。"),
        StepNotice(6, 8, "对齐项目主轴", "将作者状态严格映射到5秒窗/2.5秒步长项目轴。", "防止尾窗错位。", "对齐逐窗表和尾窗表。"),
        StepNotice(7, 8, "恢复保护状态", "在外层恢复Artifact与Boundary_Unscored。", "保持无效状态和频段功率独立。", "保护后的睡眠分期。"),
        StepNotice(8, 8, "保存睡眠结构证据", "输出逐窗、阶段、睡眠片段、转换、配置、日志和SHA256。", "支持报告和期刊重绘。", "Antila证据包。"),
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def author_spectrum_window_count(sample_count: int, sampling_rate_hz: float) -> int:
    window_samples = int(round(sampling_rate_hz)) * 5
    if sample_count <= 0 or window_samples <= 0:
        raise ValueError("样本数和采样率必须为正数")
    return 2 * int(np.ceil(sample_count / window_samples)) - 1


def compute_pooled_thresholds(
    records: tuple[tuple[AntilaFeatureArrays, np.ndarray], ...],
    *,
    theta_delta_std: float = 1.0,
    emg_std: float = 0.0,
) -> AntilaThresholds:
    pooled: dict[str, list[np.ndarray]] = {
        "theta_delta": [],
        "emg": [],
        "delta": [],
        "sigma": [],
        "gamma": [],
    }
    for features, valid_values in records:
        valid = np.asarray(valid_values, dtype=bool)
        if len(valid) != len(features.delta):
            raise ValueError("Valid mask长度与Antila特征不一致")
        theta_delta = np.divide(
            features.theta,
            features.delta,
            out=np.full(len(valid), np.nan),
            where=np.asarray(features.delta) != 0,
        )
        stacked = np.column_stack(
            [theta_delta, features.emg, features.delta, features.sigma, features.gamma]
        )
        finite = valid & np.all(np.isfinite(stacked), axis=1)
        if np.any(finite):
            pooled["theta_delta"].append(theta_delta[finite])
            pooled["emg"].append(np.asarray(features.emg)[finite])
            pooled["delta"].append(np.asarray(features.delta)[finite])
            pooled["sigma"].append(np.asarray(features.sigma)[finite])
            pooled["gamma"].append(np.asarray(features.gamma)[finite])
    if not pooled["delta"]:
        raise RuntimeError("没有可用于pooled Valid校准的有限窗口")
    values = {name: np.concatenate(parts) for name, parts in pooled.items()}
    return AntilaThresholds(
        theta_delta=float(np.mean(values["theta_delta"]) + theta_delta_std * np.std(values["theta_delta"])),
        emg=float(np.mean(values["emg"]) + emg_std * np.std(values["emg"])),
        delta=float(np.mean(values["delta"])),
        sigma=float(np.mean(values["sigma"])),
        gamma=float(np.mean(values["gamma"])),
    )


def _flatten_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(sequence, dtype=int) for sequence in sequences if len(sequence)]
    return np.concatenate(nonempty) if nonempty else np.array([], dtype=int)


def classify_with_thresholds(
    features: AntilaFeatureArrays,
    thresholds: AntilaThresholds,
    sleepy,
    *,
    dt_seconds: float = 2.5,
    pre_wake_rem_seconds: float = 30.0,
) -> np.ndarray:
    duration = np.array([[5.0, 15.0], [0.0, 5.0], [10.0, 10.0], [10.0, 10.0], [0.0, 5.0]]) / dt_seconds
    theta_delta = np.divide(features.theta, features.delta)
    sequences = {
        "high_theta": sleepy.threshold_crossing(theta_delta, thresholds.theta_delta, duration[0, 1], duration[0, 1], 1),
        "high_emg": sleepy.threshold_crossing(features.emg, thresholds.emg, duration[1, 0], duration[1, 1], 1),
        "high_delta": sleepy.threshold_crossing(features.delta, thresholds.delta, duration[2, 0], duration[2, 1], 1),
        "high_sigma": sleepy.threshold_crossing(features.sigma, thresholds.sigma, duration[3, 0], duration[3, 1], 1),
        "high_gamma": sleepy.threshold_crossing(features.gamma, thresholds.gamma, duration[4, 0], duration[4, 1], 1),
    }
    indices = {
        name: _flatten_sequences([np.arange(start, stop + 1, dtype=int) for start, stop in values])
        for name, values in sequences.items()
    }
    all_indices = np.arange(len(features.delta))
    low_emg = np.setdiff1d(all_indices, indices["high_emg"])
    low_delta = np.setdiff1d(all_indices, indices["high_delta"])
    rem_base = np.setdiff1d(np.intersect1d(indices["high_theta"], low_delta), indices["high_emg"])
    rem_sequences = sleepy.get_sequences(rem_base, duration[0, 1])
    rem = _flatten_sequences(rem_sequences)
    nrem = np.setdiff1d(indices["high_delta"], indices["high_emg"])
    nrem = np.setdiff1d(nrem, indices["high_theta"])
    nrem = _flatten_sequences(sleepy.get_sequences(nrem))
    wake = np.union1d(low_delta, indices["high_emg"])
    wake = np.setdiff1d(wake, rem)
    wake = _flatten_sequences(sleepy.get_sequences(wake))
    sigma_nrem = np.intersect1d(indices["high_sigma"], low_delta)
    sigma_nrem = np.intersect1d(sigma_nrem, low_emg)
    sigma_nrem = np.setdiff1d(sigma_nrem, rem)
    nrem = np.union1d(nrem, sigma_nrem)
    wake = np.setdiff1d(wake, sigma_nrem)
    gamma_wake = np.intersect1d(nrem, indices["high_gamma"])
    nrem = np.setdiff1d(nrem, gamma_wake)
    wake = np.union1d(wake, gamma_wake)
    nrem = np.setdiff1d(nrem, wake)
    for rem_sequence in rem_sequences:
        if len(rem_sequence) == 0 or int(rem_sequence[0]) - 1 not in wake:
            continue
        rem_start = int(rem_sequence[0])
        nrem_end = sleepy.closest_precessor(nrem, rem_start)
        if (rem_start - nrem_end) * dt_seconds < pre_wake_rem_seconds:
            new_rem = np.arange(nrem_end + 1, rem_start)
            rem = np.union1d(rem, new_rem)
            wake = np.setdiff1d(wake, new_rem)
        else:
            wake = np.union1d(wake, rem_sequence)
            rem = np.setdiff1d(rem, rem_sequence)
    states = np.zeros(len(all_indices), dtype=int)
    states[rem.astype(int)] = 1
    states[wake.astype(int)] = 2
    states[nrem.astype(int)] = 3
    return states


def validate_antila_source(
    source_root: Path,
    *,
    spec: AntilaSourceSpec = AntilaSourceSpec(),
) -> Path:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"缺少Antila本地源码目录：{root}")
    sleepy_path = root / "PySleep" / "sleepy.py"
    if not sleepy_path.is_file():
        raise FileNotFoundError(f"缺少Antila PySleep/sleepy.py：{sleepy_path}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"无法核对Antila Git commit：{root}") from error
    if commit != spec.commit:
        raise RuntimeError(f"Antila commit不匹配：{commit}")
    actual_hash = sha256_file(sleepy_path)
    if actual_hash != spec.sleepy_sha256:
        raise RuntimeError(f"Antila sleepy.py SHA256不匹配：{actual_hash}")
    return sleepy_path.resolve()


def _stage_names(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=int)
    if values.ndim != 1 or np.any((values < 0) | (values >= len(AUTHOR_STATE_LABELS))):
        raise ValueError("Antila作者状态只能是0、1、2或3")
    return AUTHOR_STATE_LABELS[values]


def align_antila_states(
    project_axis: pd.DataFrame,
    author_states: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"start_seconds", "stop_seconds", "midpoint_seconds", "phase", "status"}
    missing = sorted(required.difference(project_axis.columns))
    if missing:
        raise ValueError(f"项目窗口轴缺少字段：{', '.join(missing)}")
    unknown_status = sorted(set(project_axis["status"].astype(str)) - PROJECT_STATUSES)
    if unknown_status:
        raise ValueError(f"项目窗口轴包含未知状态：{', '.join(unknown_status)}")
    labels = _stage_names(author_states)
    if len(labels) < len(project_axis):
        raise ValueError("Antila作者窗口少于项目窗口，无法对齐")
    aligned = project_axis.copy().reset_index(drop=True)
    aligned["author_state"] = np.asarray(author_states[: len(aligned)], dtype=int)
    aligned["sleep_stage"] = labels[: len(aligned)]
    aligned["staging_method"] = "Antila"
    aligned.loc[aligned["status"].eq("Artifact"), "sleep_stage"] = "Artifact"
    aligned.loc[
        aligned["status"].eq("Boundary_Unscored"), "sleep_stage"
    ] = "Boundary_Unscored"
    tail_states = np.asarray(author_states[len(aligned) :], dtype=int)
    tail = pd.DataFrame(
        {
            "author_window_index": np.arange(len(aligned), len(author_states), dtype=int),
            "author_state": tail_states,
            "sleep_stage": _stage_names(tail_states) if len(tail_states) else np.asarray([], dtype=object),
            "reason": "作者频谱零填充尾窗，不并入项目主轴",
        }
    )
    return aligned, tail


def _load_sleepy(sleepy_path: Path):
    module_name = f"antila_sleepy_{hashlib.sha256(str(sleepy_path).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, sleepy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载Antila sleepy.py：{sleepy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sleepy_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(sleepy_path.parent):
            sys.path.pop(0)
    return module


def run_antila_staging(
    project_axis: pd.DataFrame,
    config: AntilaConfig,
    *,
    source_spec: AntilaSourceSpec = AntilaSourceSpec(),
    sleepy_module=None,
) -> AntilaStagingResult:
    errors = config.validate()
    if errors:
        raise ValueError("；".join(errors))
    if config.calibration_mode != "valid_only":
        raise NotImplementedError("pooled Valid校准由配对记录适配器执行")
    sleepy_path = validate_antila_source(config.source_root, spec=source_spec)
    sleepy = sleepy_module if sleepy_module is not None else _load_sleepy(sleepy_path)
    valid_indices = np.flatnonzero(project_axis["status"].eq("Valid").to_numpy()).tolist()
    states, details = sleepy.sleep_state(
        str(config.data_root),
        config.recording,
        th_delta_std=config.th_delta_std,
        mu_std=config.mu_std,
        sf=1,
        sf_delta=3,
        pwrite=0,
        pplot=False,
        pemg=True,
        use_idx=valid_indices,
    )
    windows, tail = align_antila_states(project_axis, np.asarray(states, dtype=int))
    return AntilaStagingResult(
        config=config,
        windows=windows,
        author_tail_windows=tail,
        source_path=sleepy_path,
        source_details=tuple(sorted(str(key) for key in details)),
    )


def write_antila_evidence(
    result: AntilaStagingResult,
    output_directory: Path,
    *,
    now: datetime,
) -> AntilaEvidenceBundle:
    directory = output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    window_path = next_versioned_path(directory, "Antila分期逐窗结果", ".csv", now)
    tail_path = next_versioned_path(directory, "Antila作者尾窗", ".csv", now)
    summary_path = next_versioned_path(directory, "Antila分期阶段汇总", ".csv", now)
    config_path = next_versioned_path(directory, "Antila分期配置", ".json", now)
    report_path = next_versioned_path(directory, "Antila分期方法报告", ".md", now)
    result.windows.to_csv(window_path, index=False, encoding="utf-8-sig")
    result.author_tail_windows.to_csv(tail_path, index=False, encoding="utf-8-sig")
    summary = (
        result.windows.groupby(["phase", "sleep_stage"], dropna=False)
        .size()
        .rename("windows")
        .reset_index()
    )
    summary["duration_seconds"] = summary["windows"] * 2.5
    phase_totals = summary.groupby("phase")["windows"].transform("sum")
    summary["percentage_of_phase"] = 100.0 * summary["windows"] / phase_totals
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    payload = {
        "config": {**asdict(result.config), "source_root": str(result.config.source_root), "data_root": str(result.config.data_root)},
        "source": {
            "repository": ANTILA_SOURCE_REPOSITORY,
            "commit": ANTILA_SOURCE_COMMIT,
            "sleepy_sha256": sha256_file(result.source_path),
            "sleepy_path": str(result.source_path),
        },
        "project_windows": len(result.windows),
        "author_tail_windows": len(result.author_tail_windows),
        "stage_counts": dict(Counter(result.windows["sleep_stage"])),
    }
    write_bytes_exclusive(config_path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_bytes_exclusive(
        report_path,
        (
            "# Antila分期方法报告\n\n"
            "- 方法：调用固定commit的作者PySleep/sleepy.py。\n"
            "- 校准：仅使用项目Valid窗。\n"
            "- 状态：Wake、NREM、REM、Uncertain；Artifact与Boundary在外层恢复。\n"
            "- 尾窗：作者额外零填充尾窗单独保存，不并入项目时间轴。\n"
        ).encode("utf-8"),
    )
    return AntilaEvidenceBundle(window_path, tail_path, summary_path, config_path, report_path)
