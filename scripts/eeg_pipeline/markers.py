"""Resolve user-supplied Markers onto the concatenated MNE time axis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from .importer import InputMetadata
from .models import MarkerSpec
from .workspace import next_versioned_path, write_bytes_exclusive


class MarkerResolutionError(RuntimeError):
    """Raised when a Marker cannot be placed without guessing or correction."""


@dataclass(frozen=True)
class ResolvedMarker:
    name: str
    onset_seconds: float
    meaning: str
    original_value: str
    original_basis: str
    source_file: str | None
    source_offset_seconds: float


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


def _parse_datetime(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MarkerResolutionError(f"{label}不是有效ISO 8601时间：{value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MarkerResolutionError(f"{label}缺少时区或UTC偏移")
    return parsed


def _source_match(
    source_file: str,
    metadata: tuple[InputMetadata, ...],
) -> InputMetadata:
    exact = [item for item in metadata if str(item.path) == source_file]
    if len(exact) == 1:
        return exact[0]
    by_name = [item for item in metadata if item.path.name == source_file]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise MarkerResolutionError(f"Marker所在文件名不唯一：{source_file}")
    raise MarkerResolutionError(f"未知Marker所在文件：{source_file}")


def _source_offset(
    target: InputMetadata,
    ordered_metadata: tuple[InputMetadata, ...],
) -> float:
    return sum(
        item.n_times / item.sfreq
        for item in ordered_metadata
        if item.order_index < target.order_index
    )


def convert_markers(
    specs: Iterable[MarkerSpec],
    input_metadata: tuple[InputMetadata, ...],
    concatenated_raw: mne.io.BaseRaw,
    *,
    expected_order: tuple[str, ...] | None = None,
) -> list[ResolvedMarker]:
    """Convert three supported timing bases without clamping or inference."""

    if not input_metadata:
        raise MarkerResolutionError("缺少输入文件元数据")
    ordered_metadata = tuple(sorted(input_metadata, key=lambda item: item.order_index))
    first = ordered_metadata[0]
    duration_seconds = concatenated_raw.n_times / float(concatenated_raw.info["sfreq"])
    resolved: list[ResolvedMarker] = []

    for spec in specs:
        validation_errors = spec.validate()
        if validation_errors:
            raise MarkerResolutionError("；".join(validation_errors))
        source_offset_seconds = 0.0
        if spec.basis == "source_relative_seconds":
            if not spec.source_file:
                raise MarkerResolutionError("源文件相对秒数缺少所在文件")
            target = _source_match(spec.source_file, ordered_metadata)
            source_offset_seconds = _source_offset(target, ordered_metadata)
            try:
                onset_seconds = source_offset_seconds + float(spec.value)
            except ValueError as error:
                raise MarkerResolutionError(
                    f"Marker时间不是秒数：{spec.name}={spec.value}"
                ) from error
        elif spec.basis == "concatenated_seconds":
            try:
                onset_seconds = float(spec.value)
            except ValueError as error:
                raise MarkerResolutionError(
                    f"Marker时间不是秒数：{spec.name}={spec.value}"
                ) from error
        elif spec.basis == "absolute_clock":
            if first.measurement_time is None:
                raise MarkerResolutionError("首个文件缺少测量时间，不能换算绝对时钟")
            first_time = _parse_datetime(first.measurement_time, "首个文件测量时间")
            marker_time = _parse_datetime(spec.value, f"Marker {spec.name}")
            onset_seconds = (marker_time - first_time).total_seconds()
        else:
            raise MarkerResolutionError(f"不支持的Marker计时方式：{spec.basis}")

        if not np.isfinite(onset_seconds) or not (
            0.0 <= onset_seconds < duration_seconds
        ):
            raise MarkerResolutionError(
                f"Marker超出合并数据范围：{spec.name}={onset_seconds} s，"
                f"数据范围=0至{duration_seconds} s"
            )
        resolved.append(
            ResolvedMarker(
                name=spec.name,
                onset_seconds=float(onset_seconds),
                meaning=spec.meaning,
                original_value=spec.value,
                original_basis=spec.basis,
                source_file=spec.source_file,
                source_offset_seconds=float(source_offset_seconds),
            )
        )

    by_name: dict[str, ResolvedMarker] = {}
    for marker in resolved:
        prior = by_name.get(marker.name)
        if prior is not None:
            if not np.isclose(prior.onset_seconds, marker.onset_seconds):
                raise MarkerResolutionError(f"同名Marker时间冲突：{marker.name}")
            raise MarkerResolutionError(f"Marker名称重复：{marker.name}")
        by_name[marker.name] = marker

    if expected_order is not None:
        missing = [name for name in expected_order if name not in by_name]
        if missing:
            raise MarkerResolutionError(
                f"预期Marker顺序中缺少事件：{', '.join(missing)}"
            )
        expected_onsets = [by_name[name].onset_seconds for name in expected_order]
        if any(
            later <= earlier
            for earlier, later in zip(expected_onsets, expected_onsets[1:])
        ):
            raise MarkerResolutionError("Marker顺序与实验日志冲突")
    return resolved


def attach_markers(
    raw: mne.io.BaseRaw,
    markers: Iterable[ResolvedMarker],
) -> mne.io.BaseRaw:
    """Return a copy with zero-duration Marker annotations appended."""

    marker_list = list(markers)
    result = raw.copy()
    if not marker_list:
        return result
    new_annotations = mne.Annotations(
        onset=[marker.onset_seconds for marker in marker_list],
        duration=[0.0] * len(marker_list),
        description=[marker.name for marker in marker_list],
        orig_time=result.annotations.orig_time,
    )
    result.set_annotations(result.annotations + new_annotations)
    return result


def plot_marker_timeline(
    markers: Iterable[ResolvedMarker],
    duration: float,
    destination: Path,
) -> Path:
    """Save a content-titled timeline without overwriting an existing image."""

    marker_list = list(markers)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    chinese_font = _chinese_font()
    axis.hlines(0.0, 0.0, duration, color="#404040", linewidth=1.5)
    colors = plt.get_cmap("tab10")
    for index, marker in enumerate(marker_list):
        color = colors(index % 10)
        axis.axvline(marker.onset_seconds, color=color, linewidth=1.8)
        axis.text(
            marker.onset_seconds,
            0.10 + 0.12 * (index % 2),
            f"{marker.name}\n{marker.onset_seconds:g} s",
            color=color,
            ha="center",
            va="bottom",
            fontsize=9,
            fontproperties=chinese_font,
        )
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-0.15, 0.42)
    axis.set_yticks([])
    axis.set_xlabel("合并后时间（秒）", fontproperties=chinese_font)
    axis.set_title("实验Marker时间轴", fontproperties=chinese_font)
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.8)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def write_marker_evidence(
    markers: Iterable[ResolvedMarker],
    annotated_raw: mne.io.BaseRaw,
    directory: Path,
    *,
    now: datetime,
) -> tuple[Path, Path, Path, Path]:
    """Save resolved Markers, annotations, report, and a timeline image."""

    marker_list = list(markers)
    marker_csv = next_versioned_path(directory, "Marker换算", ".csv", now)
    annotations_csv = next_versioned_path(directory, "Annotations", ".csv", now)
    report_path = next_versioned_path(directory, "Marker换算报告", ".md", now)
    timeline_path = next_versioned_path(directory, "实验Marker时间轴", ".png", now)

    marker_fields = [
        "name",
        "onset_seconds",
        "meaning",
        "original_value",
        "original_basis",
        "source_file",
        "source_offset_seconds",
    ]
    marker_rows = [asdict(marker) for marker in marker_list]
    annotation_fields = ["onset", "duration", "description"]
    annotation_rows = [
        {
            "onset": float(onset),
            "duration": float(duration),
            "description": str(description),
        }
        for onset, duration, description in zip(
            annotated_raw.annotations.onset,
            annotated_raw.annotations.duration,
            annotated_raw.annotations.description,
        )
    ]
    report = (
        "# Marker换算报告\n\n"
        f"- Marker数量：{len(marker_list)}\n"
        f"- 合并数据时长：{annotated_raw.n_times / annotated_raw.info['sfreq']} s\n"
        "- Marker均来自用户提供的信息，程序未猜测或自动修正时间。\n"
        "- 图表：实验Marker时间轴显示统一秒数轴和所有Marker位置。\n"
    )

    write_bytes_exclusive(marker_csv, _csv_bytes(marker_fields, marker_rows))
    write_bytes_exclusive(
        annotations_csv,
        _csv_bytes(annotation_fields, annotation_rows),
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    plot_marker_timeline(
        marker_list,
        annotated_raw.n_times / float(annotated_raw.info["sfreq"]),
        timeline_path,
    )
    return marker_csv, annotations_csv, report_path, timeline_path
