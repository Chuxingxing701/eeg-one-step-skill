from datetime import datetime, timezone
from pathlib import Path
import csv
import sys

import mne
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.importer import InputMetadata  # noqa: E402
from eeg_pipeline.markers import (  # noqa: E402
    MarkerResolutionError,
    attach_markers,
    convert_markers,
    plot_marker_timeline,
    write_marker_evidence,
)
from eeg_pipeline.models import MarkerSpec  # noqa: E402


def metadata(
    path: Path,
    *,
    order_index: int,
    n_times: int,
    sfreq: float = 100.0,
    measurement_time: str | None = "2026-01-26T11:20:00+08:00",
) -> InputMetadata:
    return InputMetadata(
        order_index=order_index,
        path=path.resolve(),
        suffix=path.suffix,
        size_bytes=100,
        modified_time="2026-07-27T17:00:00+08:00",
        channel_names=("Channel1", "Channel2", "Channel3", "Channel4"),
        channel_types=("emg", "emg", "eeg", "eeg"),
        channel_sfreqs=(sfreq, sfreq, sfreq, sfreq),
        sfreq=sfreq,
        duration_seconds=n_times / sfreq,
        n_times=n_times,
        measurement_time=measurement_time,
        sha256="0" * 64,
    )


def make_raw(duration_seconds: float = 30.0, sfreq: float = 100.0):
    n_times = int(duration_seconds * sfreq)
    data = np.zeros((2, n_times), dtype=float)
    info = mne.create_info(
        ["EEG_bipolar", "EMG_bipolar"],
        sfreq=sfreq,
        ch_types=["eeg", "emg"],
    )
    return mne.io.RawArray(data, info, verbose="ERROR")


def two_segments(tmp_path: Path) -> tuple[InputMetadata, InputMetadata]:
    return (
        metadata(tmp_path / "mouse_6.edf", order_index=0, n_times=1_000),
        metadata(tmp_path / "mouse_7.edf", order_index=1, n_times=2_000),
    )


def test_source_relative_seconds_add_exact_preceding_sample_duration(
    tmp_path: Path,
) -> None:
    specs = [
        MarkerSpec(
            "Recovery_start",
            "5",
            "source_relative_seconds",
            "mouse_7.edf",
            "恢复期开始",
        )
    ]

    result = convert_markers(specs, two_segments(tmp_path), make_raw())

    assert result[0].onset_seconds == 15.0
    assert result[0].source_offset_seconds == 10.0


def test_concatenated_seconds_remain_unchanged(tmp_path: Path) -> None:
    specs = [
        MarkerSpec(
            "N2O_start",
            "12.5",
            "concatenated_seconds",
            None,
            "开始吸入N2O",
        )
    ]

    result = convert_markers(specs, two_segments(tmp_path), make_raw())

    assert result[0].onset_seconds == 12.5
    assert result[0].source_offset_seconds == 0.0


def test_absolute_clock_subtracts_first_measurement_time(tmp_path: Path) -> None:
    specs = [
        MarkerSpec(
            "Clock_marker",
            "2026-01-26T11:20:10+08:00",
            "absolute_clock",
            None,
            "绝对时钟事件",
        )
    ]

    result = convert_markers(specs, two_segments(tmp_path), make_raw())

    assert result[0].onset_seconds == 10.0


def test_unknown_source_file_blocks_conversion(tmp_path: Path) -> None:
    specs = [
        MarkerSpec(
            "N2O_start",
            "1",
            "source_relative_seconds",
            "mouse_8.edf",
            "开始吸入N2O",
        )
    ]

    with pytest.raises(MarkerResolutionError, match="未知Marker所在文件"):
        convert_markers(specs, two_segments(tmp_path), make_raw())


@pytest.mark.parametrize("value", ["-0.1", "30", "31"])
def test_out_of_bounds_marker_is_not_clamped(tmp_path: Path, value: str) -> None:
    specs = [
        MarkerSpec(
            "Out_of_bounds",
            value,
            "concatenated_seconds",
            None,
            "越界事件",
        )
    ]

    with pytest.raises(MarkerResolutionError, match="超出合并数据范围"):
        convert_markers(specs, two_segments(tmp_path), make_raw())


def test_duplicate_name_with_conflicting_time_blocks_conversion(
    tmp_path: Path,
) -> None:
    specs = [
        MarkerSpec("N2O_start", "10", "concatenated_seconds", None, "事件"),
        MarkerSpec("N2O_start", "11", "concatenated_seconds", None, "事件"),
    ]

    with pytest.raises(MarkerResolutionError, match="同名Marker时间冲突"):
        convert_markers(specs, two_segments(tmp_path), make_raw())


def test_expected_experiment_order_must_increase(tmp_path: Path) -> None:
    specs = [
        MarkerSpec("Box_start", "20", "concatenated_seconds", None, "适应期"),
        MarkerSpec("N2O_start", "10", "concatenated_seconds", None, "给药期"),
    ]

    with pytest.raises(MarkerResolutionError, match="Marker顺序与实验日志冲突"):
        convert_markers(
            specs,
            two_segments(tmp_path),
            make_raw(),
            expected_order=("Box_start", "N2O_start"),
        )


def test_absolute_clock_requires_first_measurement_time(tmp_path: Path) -> None:
    items = list(two_segments(tmp_path))
    items[0] = metadata(
        tmp_path / "mouse_6.edf",
        order_index=0,
        n_times=1_000,
        measurement_time=None,
    )
    specs = [
        MarkerSpec(
            "Clock_marker",
            "2026-01-26T11:20:10+08:00",
            "absolute_clock",
            None,
            "绝对时钟事件",
        )
    ]

    with pytest.raises(MarkerResolutionError, match="首个文件缺少测量时间"):
        convert_markers(specs, tuple(items), make_raw())


def test_attach_markers_preserves_existing_bad_and_adds_zero_duration_annotations(
    tmp_path: Path,
) -> None:
    raw = make_raw()
    raw.set_annotations(
        mne.Annotations(onset=[2.0], duration=[1.0], description=["BAD_existing"])
    )
    specs = [
        MarkerSpec("N2O_start", "10", "concatenated_seconds", None, "给药期")
    ]
    resolved = convert_markers(specs, two_segments(tmp_path), raw)

    result = attach_markers(raw, resolved)

    assert list(raw.annotations.description) == ["BAD_existing"]
    assert list(result.annotations.description) == ["BAD_existing", "N2O_start"]
    marker_index = list(result.annotations.description).index("N2O_start")
    assert result.annotations.duration[marker_index] == 0.0
    assert result.annotations.onset[marker_index] == 10.0


def test_marker_timeline_and_evidence_outputs_are_content_named(
    tmp_path: Path,
) -> None:
    raw = make_raw()
    specs = [
        MarkerSpec("Box_start", "5", "concatenated_seconds", None, "适应期"),
        MarkerSpec("N2O_start", "10", "concatenated_seconds", None, "给药期"),
        MarkerSpec("Recovery_start", "20", "concatenated_seconds", None, "恢复期"),
    ]
    resolved = convert_markers(
        specs,
        two_segments(tmp_path),
        raw,
        expected_order=("Box_start", "N2O_start", "Recovery_start"),
    )
    destination = tmp_path / "实验Marker时间轴-20260727-1730-ver.001.png"

    plot_marker_timeline(resolved, 30.0, destination)
    outputs = write_marker_evidence(
        resolved,
        attach_markers(raw, resolved),
        tmp_path / "evidence",
        now=datetime(2026, 7, 27, 17, 30),
    )

    assert destination.exists() and destination.stat().st_size > 1_000
    assert "实验Marker时间轴" in destination.name
    assert len(outputs) == 4
    assert {path.suffix for path in outputs} == {".csv", ".md", ".png"}
    marker_csv = next(path for path in outputs if "Marker换算" in path.name)
    with marker_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["name"] for row in rows] == [
        "Box_start",
        "N2O_start",
        "Recovery_start",
    ]
    assert [float(row["onset_seconds"]) for row in rows] == [5.0, 10.0, 20.0]
