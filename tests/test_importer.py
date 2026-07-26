from datetime import datetime
from pathlib import Path
import sys

import mne
import numpy as np
from numpy.testing import assert_allclose
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.importer import (  # noqa: E402
    InputCompatibilityError,
    assert_compatible,
    import_and_concatenate,
    inspect_inputs,
    write_import_evidence,
)
from eeg_pipeline.models import BipolarPair, MarkerSpec, RunConfig  # noqa: E402
from eeg_pipeline.provenance import sha256_file  # noqa: E402


CHANNEL_NAMES = ("Channel1", "Channel2", "Channel3", "Channel4")
CHANNEL_TYPES = ("emg", "emg", "eeg", "eeg")


def make_raw(
    *,
    sfreq: float = 100.0,
    n_times: int = 200,
    offset: float = 0.0,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
) -> tuple[mne.io.RawArray, np.ndarray]:
    channel_types = tuple(
        CHANNEL_TYPES[CHANNEL_NAMES.index(name)]
        for name in channel_names
        if name in CHANNEL_NAMES
    )
    sample_axis = np.arange(n_times, dtype=float) + offset
    data = np.vstack(
        [sample_axis + channel_index * 1_000 for channel_index in range(len(channel_names))]
    ) * 1e-6
    info = mne.create_info(
        list(channel_names),
        sfreq=sfreq,
        ch_types=list(channel_types),
    )
    return mne.io.RawArray(data, info, verbose="ERROR"), data


def save_fif(
    path: Path,
    *,
    sfreq: float = 100.0,
    n_times: int = 200,
    offset: float = 0.0,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
) -> np.ndarray:
    raw, data = make_raw(
        sfreq=sfreq,
        n_times=n_times,
        offset=offset,
        channel_names=channel_names,
    )
    raw.save(path, overwrite=False, verbose="ERROR")
    return data


def make_config(tmp_path: Path, input_files: tuple[Path, ...], sfreq=100.0) -> RunConfig:
    return RunConfig(
        output_root=tmp_path / "results",
        input_files=input_files,
        expected_sfreq=sfreq,
        file_order_confirmed=True,
        same_animal_continuous_recording=True,
        bipolar_pairs=(
            BipolarPair("EEG_bipolar", "Channel4", "Channel3", "eeg"),
            BipolarPair("EMG_bipolar", "Channel2", "Channel1", "emg"),
        ),
        markers=(
            MarkerSpec(
                "N2O_start",
                "1.0",
                "concatenated_seconds",
                None,
                "开始吸入N2O",
            ),
        ),
    )


def test_inspect_fif_records_header_and_sha256_without_preloading(tmp_path: Path) -> None:
    source = tmp_path / "segment_1_raw.fif"
    save_fif(source)

    metadata = inspect_inputs(make_config(tmp_path, (source,)))

    assert len(metadata) == 1
    item = metadata[0]
    assert item.path == source.resolve()
    assert item.suffix == ".fif"
    assert item.channel_names == CHANNEL_NAMES
    assert item.channel_types == CHANNEL_TYPES
    assert item.channel_sfreqs == (100.0, 100.0, 100.0, 100.0)
    assert item.sfreq == 100.0
    assert item.n_times == 200
    assert item.duration_seconds == 2.0
    assert item.sha256 == sha256_file(source)
    assert item.size_bytes == source.stat().st_size


def test_inspect_edf_uses_real_mne_edf_reader(tmp_path: Path) -> None:
    source = tmp_path / "segment_1.edf"
    raw, _ = make_raw(n_times=200)
    raw.export(source, fmt="edf", physical_range="auto", overwrite=False)

    metadata = inspect_inputs(make_config(tmp_path, (source,)))

    assert metadata[0].suffix == ".edf"
    assert metadata[0].channel_names == CHANNEL_NAMES
    assert metadata[0].sfreq == 100.0
    assert metadata[0].n_times == 200


def test_direct_concatenation_preserves_order_and_every_sample(tmp_path: Path) -> None:
    first = tmp_path / "segment_1_raw.fif"
    second = tmp_path / "segment_2_raw.fif"
    first_data = save_fif(first, n_times=200, offset=0)
    second_data = save_fif(second, n_times=300, offset=10_000)
    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    destination = tmp_path / "直接拼接后数据-20260727-1700-ver.001_raw.fif"

    result = import_and_concatenate(
        make_config(tmp_path, (first, second)),
        destination,
    )

    merged = mne.io.read_raw_fif(destination, preload=True, verbose="ERROR")
    assert result.source_n_times == (200, 300)
    assert result.n_times == 500
    assert merged.n_times == 500
    assert_allclose(merged.get_data()[:, :200], first_data, rtol=1e-6, atol=1e-12)
    assert_allclose(merged.get_data()[:, 200:], second_data, rtol=1e-6, atol=1e-12)
    assert_allclose(merged.get_data()[:, 199], first_data[:, -1], rtol=1e-6, atol=1e-12)
    assert_allclose(merged.get_data()[:, 200], second_data[:, 0], rtol=1e-6, atol=1e-12)
    assert result.direct_concatenation_assumption is True
    assert "未执行重叠/缺口检测" in result.method_statement
    assert sha256_file(first) == first_hash
    assert sha256_file(second) == second_hash


def test_single_fif_is_saved_without_concatenation_or_trimming(tmp_path: Path) -> None:
    source = tmp_path / "single_raw.fif"
    expected = save_fif(source, n_times=250)
    destination = tmp_path / "直接导入后数据-20260727-1700-ver.001_raw.fif"

    result = import_and_concatenate(make_config(tmp_path, (source,)), destination)

    reopened = mne.io.read_raw_fif(destination, preload=True, verbose="ERROR")
    assert result.source_n_times == (250,)
    assert reopened.n_times == 250
    assert_allclose(reopened.get_data(), expected, rtol=1e-6, atol=1e-12)


def test_missing_required_channel_blocks_import(tmp_path: Path) -> None:
    source = tmp_path / "missing_channel_raw.fif"
    save_fif(source, channel_names=("Channel1", "Channel2", "Channel3"))

    with pytest.raises(InputCompatibilityError, match="缺少必需通道.*Channel4"):
        inspect_inputs(make_config(tmp_path, (source,)))


def test_mismatched_sampling_rates_block_concatenation(tmp_path: Path) -> None:
    first = tmp_path / "first_raw.fif"
    second = tmp_path / "second_raw.fif"
    save_fif(first, sfreq=100.0)
    save_fif(second, sfreq=128.0)
    config = make_config(tmp_path, (first, second), sfreq=None)

    with pytest.raises(InputCompatibilityError, match="采样率不兼容"):
        inspect_inputs(config)


def test_reordered_channel_structure_blocks_concatenation(tmp_path: Path) -> None:
    first = tmp_path / "first_raw.fif"
    second = tmp_path / "second_raw.fif"
    save_fif(first)
    save_fif(
        second,
        channel_names=("Channel2", "Channel1", "Channel3", "Channel4"),
    )
    config = make_config(tmp_path, (first, second))

    with pytest.raises(InputCompatibilityError, match="通道顺序或结构不兼容"):
        inspect_inputs(config)


def test_unreadable_input_path_stops_before_concatenation(tmp_path: Path) -> None:
    missing = tmp_path / "missing_raw.fif"

    with pytest.raises(FileNotFoundError):
        inspect_inputs(make_config(tmp_path, (missing,)))


def test_assert_compatible_rejects_empty_metadata() -> None:
    with pytest.raises(InputCompatibilityError, match="没有可导入的文件"):
        assert_compatible((), {"Channel1"})


def test_import_evidence_reports_direct_concat_assumption(tmp_path: Path) -> None:
    first = tmp_path / "first_raw.fif"
    second = tmp_path / "second_raw.fif"
    save_fif(first)
    save_fif(second, offset=10_000)
    destination = tmp_path / "merged_raw.fif"
    result = import_and_concatenate(
        make_config(tmp_path, (first, second)),
        destination,
    )

    outputs = write_import_evidence(
        result,
        tmp_path / "evidence",
        now=datetime(2026, 7, 27, 17, 5),
    )

    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    report = next(path for path in outputs if path.suffix == ".md")
    text = report.read_text(encoding="utf-8")
    assert "未执行重叠/缺口检测" in text
    assert "输入样本数总和：400" in text
    assert "输出样本数：400" in text
