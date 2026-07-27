"""Read EDF/FIF headers and concatenate files in user-confirmed order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json

import mne
import numpy as np

from .models import RunConfig
from .provenance import sha256_file
from .workspace import next_versioned_path, write_bytes_exclusive


class InputCompatibilityError(RuntimeError):
    """Raised when selected files cannot enter direct concatenation."""


@dataclass(frozen=True)
class InputMetadata:
    order_index: int
    path: Path
    suffix: str
    size_bytes: int
    modified_time: str
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    channel_sfreqs: tuple[float, ...]
    sfreq: float
    duration_seconds: float
    n_times: int
    measurement_time: str | None
    sha256: str


@dataclass(frozen=True)
class ImportResult:
    destination: Path
    destination_sha256: str
    inputs: tuple[InputMetadata, ...]
    source_n_times: tuple[int, ...]
    selected_channel_names: tuple[str, ...]
    selected_channel_types: tuple[str, ...]
    excluded_channel_names: tuple[str, ...]
    n_times: int
    sfreq: float
    duration_seconds: float
    direct_concatenation_assumption: bool
    method_statement: str


def _read_raw(path: Path, *, preload: bool) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    raise ValueError(f"不支持的输入格式：{path.suffix}")


def _measurement_time(raw: mne.io.BaseRaw) -> str | None:
    value = raw.info.get("meas_date")
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def inspect_inputs(config: RunConfig) -> tuple[InputMetadata, ...]:
    """Read headers and hashes without changing file order or loading signals."""

    config_errors = config.validate()
    if config_errors:
        raise ValueError("；".join(config_errors))

    metadata: list[InputMetadata] = []
    for order_index, configured_path in enumerate(config.input_files):
        path = configured_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = _read_raw(path, preload=False)
        try:
            sfreq = float(raw.info["sfreq"])
            item = InputMetadata(
                order_index=order_index,
                path=path,
                suffix=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                modified_time=datetime.fromtimestamp(
                    path.stat().st_mtime
                ).astimezone().isoformat(),
                channel_names=tuple(raw.ch_names),
                channel_types=tuple(raw.get_channel_types()),
                channel_sfreqs=tuple(sfreq for _ in raw.ch_names),
                sfreq=sfreq,
                duration_seconds=raw.n_times / sfreq,
                n_times=int(raw.n_times),
                measurement_time=_measurement_time(raw),
                sha256=sha256_file(path),
            )
        finally:
            raw.close()
        metadata.append(item)

    required_channels = {
        channel
        for pair in config.bipolar_pairs
        for channel in (pair.positive_channel, pair.reference_channel)
    }
    result = tuple(metadata)
    assert_compatible(result, required_channels)
    if config.expected_sfreq is not None:
        for item in result:
            if not np.isclose(item.sfreq, config.expected_sfreq):
                raise InputCompatibilityError(
                    f"读取采样率与Checklist不一致：{item.path.name}="
                    f"{item.sfreq} Hz，预期={config.expected_sfreq} Hz"
                )
    return result


def assert_compatible(
    metadata: tuple[InputMetadata, ...],
    required_channels: set[str],
) -> None:
    """Require equal sampling rate and exact channel structure in every file."""

    if not metadata:
        raise InputCompatibilityError("没有可导入的文件")
    reference = metadata[0]
    for item in metadata:
        missing = sorted(required_channels.difference(item.channel_names))
        if missing:
            raise InputCompatibilityError(
                f"{item.path.name}缺少必需通道：{', '.join(missing)}"
            )
        if not np.isclose(item.sfreq, reference.sfreq):
            raise InputCompatibilityError(
                f"采样率不兼容：{reference.path.name}={reference.sfreq} Hz，"
                f"{item.path.name}={item.sfreq} Hz"
            )
        if (
            item.channel_names != reference.channel_names
            or item.channel_types != reference.channel_types
        ):
            raise InputCompatibilityError(
                f"通道顺序或结构不兼容：{item.path.name}"
            )


def _selected_channel_names(config: RunConfig, source_names: tuple[str, ...]) -> tuple[str, ...]:
    required = {
        channel
        for pair in config.bipolar_pairs
        for channel in (pair.positive_channel, pair.reference_channel)
    }
    return tuple(name for name in source_names if name in required)


def _selected_channel_type_map(config: RunConfig) -> dict[str, str]:
    channel_types: dict[str, str] = {}
    for pair in config.bipolar_pairs:
        for channel in (pair.positive_channel, pair.reference_channel):
            existing = channel_types.get(channel)
            if existing is not None and existing != pair.channel_type:
                raise ValueError(
                    f"同一分析通道被配置为冲突类型：{channel}={existing}/{pair.channel_type}"
                )
            channel_types[channel] = pair.channel_type
    return channel_types


def import_and_concatenate(config: RunConfig, destination: Path) -> ImportResult:
    """Save one direct-concatenation FIF without overlap/gap inference."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    metadata = inspect_inputs(config)
    selected_channel_names = _selected_channel_names(config, metadata[0].channel_names)
    selected_channel_type_map = _selected_channel_type_map(config)
    selected_channel_types = tuple(
        selected_channel_type_map[name] for name in selected_channel_names
    )
    excluded_channel_names = tuple(
        name for name in metadata[0].channel_names if name not in selected_channel_names
    )
    raws = [_read_raw(item.path, preload=False) for item in metadata]
    for raw in raws:
        raw.pick(list(selected_channel_names))
        raw.set_channel_types(selected_channel_type_map, on_unit_change="ignore", verbose="ERROR")
    merged: mne.io.BaseRaw
    try:
        if len(raws) == 1:
            merged = raws[0]
        else:
            merged = mne.concatenate_raws(
                raws,
                preload=False,
                on_mismatch="raise",
                verbose="ERROR",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        merged.save(destination, overwrite=False, verbose="ERROR")
        n_times = int(merged.n_times)
        sfreq = float(merged.info["sfreq"])
    finally:
        for raw in raws:
            raw.close()

    source_n_times = tuple(item.n_times for item in metadata)
    if n_times != sum(source_n_times):
        raise RuntimeError(
            f"直接拼接样本数不守恒：输出={n_times}，输入总和={sum(source_n_times)}"
        )
    return ImportResult(
        destination=destination,
        destination_sha256=sha256_file(destination),
        inputs=metadata,
        source_n_times=source_n_times,
        selected_channel_names=selected_channel_names,
        selected_channel_types=selected_channel_types,
        excluded_channel_names=excluded_channel_names,
        n_times=n_times,
        sfreq=sfreq,
        duration_seconds=n_times / sfreq,
        direct_concatenation_assumption=True,
        method_statement=(
            "文件按用户确认顺序直接拼接；未执行重叠/缺口检测；"
            "只保留双极公式涉及的用户确认分析通道并应用确认的EEG/EMG类型；"
            "未删除、移动或补造任何样本。"
        ),
    )


def _metadata_payload(item: InputMetadata) -> dict:
    payload = asdict(item)
    payload["path"] = str(item.path)
    return payload


def write_import_evidence(
    result: ImportResult,
    directory: Path,
    *,
    now: datetime,
) -> tuple[Path, Path]:
    """Write versioned machine-readable metadata and a plain-language report."""

    json_path = next_versioned_path(
        directory,
        "输入文件与直接拼接报告",
        ".json",
        now,
    )
    markdown_path = next_versioned_path(
        directory,
        "输入文件与直接拼接报告",
        ".md",
        now,
    )
    payload = {
        "destination": str(result.destination),
        "destination_sha256": result.destination_sha256,
        "inputs": [_metadata_payload(item) for item in result.inputs],
        "source_n_times": list(result.source_n_times),
        "selected_channel_names": list(result.selected_channel_names),
        "selected_channel_types": list(result.selected_channel_types),
        "excluded_channel_names": list(result.excluded_channel_names),
        "n_times": result.n_times,
        "sfreq": result.sfreq,
        "duration_seconds": result.duration_seconds,
        "direct_concatenation_assumption": result.direct_concatenation_assumption,
        "method_statement": result.method_statement,
    }
    report = (
        "# 输入文件与直接拼接报告\n\n"
        f"- 输入文件数：{len(result.inputs)}\n"
        f"- 输入样本数总和：{sum(result.source_n_times)}\n"
        f"- 输出样本数：{result.n_times}\n"
        f"- 保留分析通道：{', '.join(result.selected_channel_names)}\n"
        f"- 分析通道类型：{', '.join(result.selected_channel_types)}\n"
        f"- 排除非分析通道：{', '.join(result.excluded_channel_names) or '无'}\n"
        f"- 采样率：{result.sfreq} Hz\n"
        f"- 输出时长：{result.duration_seconds} s\n"
        f"- 方法：{result.method_statement}\n"
        f"- 输出SHA256：{result.destination_sha256}\n"
    )
    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    write_bytes_exclusive(json_path, json_bytes)
    write_bytes_exclusive(markdown_path, report.encode("utf-8"))
    return json_path, markdown_path
