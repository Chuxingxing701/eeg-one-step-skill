"""Apply user-confirmed bipolar references and freeze immutable FIF checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
import csv
import io
import json
import os
import sys

import mne
import numpy as np

from .models import BipolarPair
from .provenance import sha256_file
from .workspace import next_versioned_path, write_bytes_exclusive


@dataclass(frozen=True)
class BipolarQC:
    passed: bool
    errors: tuple[str, ...]
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    n_times: int
    sfreq: float
    annotations_equal: bool
    max_abs_error: dict[str, float]


@dataclass(frozen=True)
class FrozenRaw:
    path: Path
    sha256: str
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    n_times: int
    sfreq: float
    reopen_qc_passed: bool
    evidence_paths: tuple[Path, ...]


def _annotations_equal(left: mne.Annotations, right: mne.Annotations) -> bool:
    return (
        left.orig_time == right.orig_time
        and np.array_equal(left.description, right.description)
        and np.allclose(left.onset, right.onset, rtol=0.0, atol=1e-12)
        and np.allclose(left.duration, right.duration, rtol=0.0, atol=1e-12)
    )


def apply_bipolar_reference(
    raw: mne.io.BaseRaw,
    pairs: tuple[BipolarPair, ...],
) -> mne.io.BaseRaw:
    """Return only requested bipolar channels while leaving source untouched."""

    if not pairs:
        raise ValueError("必须提供至少一条双极参考公式")
    errors = [error for pair in pairs for error in pair.validate()]
    if errors:
        raise ValueError("；".join(dict.fromkeys(errors)))
    output_names = [pair.output_name for pair in pairs]
    if len(output_names) != len(set(output_names)):
        raise ValueError("双极输出通道名不得重复")
    source_channels = {
        channel
        for pair in pairs
        for channel in (pair.positive_channel, pair.reference_channel)
    }
    missing = sorted(source_channels.difference(raw.ch_names))
    if missing:
        raise ValueError(f"缺少双极参考源通道：{', '.join(missing)}")
    collisions = sorted(set(output_names).intersection(raw.ch_names))
    if collisions:
        raise ValueError(f"双极输出通道名与原通道冲突：{', '.join(collisions)}")

    working = raw.copy().load_data()
    result = mne.set_bipolar_reference(
        working,
        anode=[pair.positive_channel for pair in pairs],
        cathode=[pair.reference_channel for pair in pairs],
        ch_name=output_names,
        drop_refs=True,
        copy=False,
        on_bad="warn",
        verbose="ERROR",
    )
    result.pick(output_names)
    result.set_channel_types(
        {pair.output_name: pair.channel_type for pair in pairs},
        verbose="ERROR",
    )
    return result


def validate_bipolar_output(
    source: mne.io.BaseRaw,
    result: mne.io.BaseRaw,
    pairs: tuple[BipolarPair, ...],
    *,
    chunk_samples: int = 1_000_000,
    atol: float = 1e-12,
) -> BipolarQC:
    """Validate structure and subtraction in chunks to control memory use."""

    if chunk_samples <= 0:
        raise ValueError("chunk_samples必须为正整数")
    errors: list[str] = []
    expected_names = tuple(pair.output_name for pair in pairs)
    expected_types = tuple(pair.channel_type for pair in pairs)
    actual_names = tuple(result.ch_names)
    actual_types = tuple(result.get_channel_types())
    if actual_names != expected_names:
        errors.append("双极输出通道名称或顺序不正确")
    if actual_types != expected_types:
        errors.append("双极输出通道类型不正确")
    if result.n_times != source.n_times:
        errors.append("双极参考后样本数发生变化")
    if not np.isclose(result.info["sfreq"], source.info["sfreq"]):
        errors.append("双极参考后采样率发生变化")
    annotations_equal = _annotations_equal(source.annotations, result.annotations)
    if not annotations_equal:
        errors.append("双极参考后Annotations发生变化")

    max_abs_error: dict[str, float] = {pair.output_name: 0.0 for pair in pairs}
    if actual_names == expected_names and result.n_times == source.n_times:
        for start in range(0, source.n_times, chunk_samples):
            stop = min(start + chunk_samples, source.n_times)
            for pair in pairs:
                source_data = source.get_data(
                    picks=[pair.positive_channel, pair.reference_channel],
                    start=start,
                    stop=stop,
                )
                actual = result.get_data(
                    picks=[pair.output_name],
                    start=start,
                    stop=stop,
                )[0]
                expected = source_data[0] - source_data[1]
                chunk_error = float(np.max(np.abs(actual - expected), initial=0.0))
                max_abs_error[pair.output_name] = max(
                    max_abs_error[pair.output_name],
                    chunk_error,
                )
        for pair in pairs:
            if max_abs_error[pair.output_name] > atol:
                errors.append(
                    f"{pair.output_name}数值不等于正通道减参考通道"
                )

    unique_errors = tuple(dict.fromkeys(errors))
    return BipolarQC(
        passed=not unique_errors,
        errors=unique_errors,
        channel_names=actual_names,
        channel_types=actual_types,
        n_times=int(result.n_times),
        sfreq=float(result.info["sfreq"]),
        annotations_equal=annotations_equal,
        max_abs_error=max_abs_error,
    )


def _raw_structure_equal(left: mne.io.BaseRaw, right: mne.io.BaseRaw) -> bool:
    return (
        left.ch_names == right.ch_names
        and left.get_channel_types() == right.get_channel_types()
        and left.n_times == right.n_times
        and np.isclose(left.info["sfreq"], right.info["sfreq"])
        and _annotations_equal(left.annotations, right.annotations)
    )


def _annotations_csv(raw: mne.io.BaseRaw) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["onset", "duration", "description"],
    )
    writer.writeheader()
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        writer.writerow(
            {
                "onset": float(onset),
                "duration": float(duration),
                "description": str(description),
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def _software_versions() -> dict[str, str]:
    packages = ("mne", "numpy", "scipy", "pandas", "matplotlib")
    versions = {name: metadata.version(name) for name in packages}
    versions["python"] = sys.version.split()[0]
    return versions


def _write_freeze_evidence(
    raw: mne.io.BaseRaw,
    path: Path,
    sha256: str,
    reopen_qc_passed: bool,
    evidence_directory: Path,
    now: datetime,
) -> tuple[Path, ...]:
    annotations_path = next_versioned_path(
        evidence_directory,
        "冻结Annotations",
        ".csv",
        now,
    )
    config_path = next_versioned_path(
        evidence_directory,
        "冻结配置",
        ".json",
        now,
    )
    report_path = next_versioned_path(
        evidence_directory,
        "冻结报告",
        ".md",
        now,
    )
    hash_path = next_versioned_path(
        evidence_directory,
        "冻结SHA256",
        ".txt",
        now,
    )
    config = {
        "path": str(path),
        "sha256": sha256,
        "channel_names": list(raw.ch_names),
        "channel_types": raw.get_channel_types(),
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_seconds": raw.n_times / float(raw.info["sfreq"]),
        "annotation_count": len(raw.annotations),
        "reopen_qc_passed": reopen_qc_passed,
        "software_versions": _software_versions(),
    }
    report = (
        "# FIF冻结报告\n\n"
        f"- 文件：{path}\n"
        f"- 通道：{', '.join(raw.ch_names)}\n"
        f"- 采样率：{raw.info['sfreq']} Hz\n"
        f"- 样本数：{raw.n_times}\n"
        f"- Annotations数量：{len(raw.annotations)}\n"
        f"- 重新打开结构QC：{'通过' if reopen_qc_passed else '失败'}\n"
        f"- SHA256：{sha256}\n"
    )
    write_bytes_exclusive(annotations_path, _annotations_csv(raw))
    write_bytes_exclusive(
        config_path,
        (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    write_bytes_exclusive(hash_path, f"{sha256}  {path.name}\n".encode("utf-8"))
    return annotations_path, config_path, report_path, hash_path


def freeze_raw(
    raw: mne.io.BaseRaw,
    destination: Path,
    evidence_directory: Path,
    *,
    now: datetime,
) -> FrozenRaw:
    """Save one non-overwriting FIF and verify it after reopening."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw.save(destination, overwrite=False, verbose="ERROR")
    frozen_hash = sha256_file(destination)
    reopened = mne.io.read_raw_fif(destination, preload=False, verbose="ERROR")
    try:
        reopen_qc_passed = _raw_structure_equal(raw, reopened)
        evidence_paths = _write_freeze_evidence(
            reopened,
            destination,
            frozen_hash,
            reopen_qc_passed,
            evidence_directory,
            now,
        )
        frozen = FrozenRaw(
            path=destination,
            sha256=frozen_hash,
            channel_names=tuple(reopened.ch_names),
            channel_types=tuple(reopened.get_channel_types()),
            n_times=int(reopened.n_times),
            sfreq=float(reopened.info["sfreq"]),
            reopen_qc_passed=reopen_qc_passed,
            evidence_paths=evidence_paths,
        )
    finally:
        reopened.close()
    if not reopen_qc_passed:
        raise RuntimeError("冻结FIF重新打开后的结构QC失败")
    return frozen


def _copy_exclusive(source: Path, destination: Path, chunk_size: int = 8_388_608) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while chunk := input_stream.read(chunk_size):
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def copy_frozen_checkpoint(
    frozen: FrozenRaw,
    destination: Path,
    evidence_directory: Path,
    *,
    now: datetime,
) -> FrozenRaw:
    """Create a byte-identical pre-artifact checkpoint without overwriting."""

    source = frozen.path.resolve(strict=True)
    if sha256_file(source) != frozen.sha256:
        raise RuntimeError("源冻结FIF的SHA256已变化")
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    _copy_exclusive(source, destination)
    copied_hash = sha256_file(destination)
    if copied_hash != frozen.sha256:
        raise RuntimeError("去伪迹前副本与双极冻结版SHA256不一致")

    source_raw = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")
    copied_raw = mne.io.read_raw_fif(destination, preload=False, verbose="ERROR")
    try:
        reopen_qc_passed = _raw_structure_equal(source_raw, copied_raw)
        evidence_paths = _write_freeze_evidence(
            copied_raw,
            destination,
            copied_hash,
            reopen_qc_passed,
            evidence_directory,
            now,
        )
        result = FrozenRaw(
            path=destination,
            sha256=copied_hash,
            channel_names=tuple(copied_raw.ch_names),
            channel_types=tuple(copied_raw.get_channel_types()),
            n_times=int(copied_raw.n_times),
            sfreq=float(copied_raw.info["sfreq"]),
            reopen_qc_passed=reopen_qc_passed,
            evidence_paths=evidence_paths,
        )
    finally:
        source_raw.close()
        copied_raw.close()
    if not reopen_qc_passed:
        raise RuntimeError("去伪迹前副本重新打开后的结构QC失败")
    return result
