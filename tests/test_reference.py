from datetime import datetime
import json
from pathlib import Path
import sys

import mne
import numpy as np
from numpy.testing import assert_allclose
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.models import BipolarPair  # noqa: E402
from eeg_pipeline.provenance import sha256_file  # noqa: E402
from eeg_pipeline.reference import (  # noqa: E402
    apply_bipolar_reference,
    copy_frozen_checkpoint,
    freeze_raw,
    validate_bipolar_output,
)


def make_four_channel_raw(sfreq: float = 500.0, n_times: int = 2_000):
    times = np.arange(n_times, dtype=float) / sfreq
    data = np.vstack(
        [
            10e-6 * np.sin(2 * np.pi * 20 * times),
            30e-6 * np.sin(2 * np.pi * 20 * times) + 2e-6,
            20e-6 * np.sin(2 * np.pi * 4 * times),
            70e-6 * np.sin(2 * np.pi * 4 * times) - 3e-6,
        ]
    )
    info = mne.create_info(
        ["Channel1", "Channel2", "Channel3", "Channel4"],
        sfreq=sfreq,
        ch_types=["emg", "emg", "eeg", "eeg"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 1.0],
            duration=[0.0, 0.25],
            description=["N2O_start", "BAD_existing"],
        )
    )
    return raw, data


def pairs() -> tuple[BipolarPair, ...]:
    return (
        BipolarPair("EEG_bipolar", "Channel4", "Channel3", "eeg"),
        BipolarPair("EMG_bipolar", "Channel2", "Channel1", "emg"),
    )


def annotation_snapshot(raw: mne.io.BaseRaw) -> tuple:
    return (
        tuple(raw.annotations.onset),
        tuple(raw.annotations.duration),
        tuple(raw.annotations.description),
        raw.annotations.orig_time,
    )


def test_apply_bipolar_reference_matches_positive_minus_reference() -> None:
    source, source_data = make_four_channel_raw()
    source_before = source.get_data().copy()
    annotations_before = annotation_snapshot(source)

    result = apply_bipolar_reference(source, pairs())

    assert result.ch_names == ["EEG_bipolar", "EMG_bipolar"]
    assert result.get_channel_types() == ["eeg", "emg"]
    expected = np.vstack(
        [
            source_data[3] - source_data[2],
            source_data[1] - source_data[0],
        ]
    )
    assert_allclose(result.get_data(), expected, rtol=1e-12, atol=1e-15)
    assert_allclose(source.get_data(), source_before, rtol=0, atol=0)
    assert source.ch_names == ["Channel1", "Channel2", "Channel3", "Channel4"]
    assert annotation_snapshot(source) == annotations_before
    assert annotation_snapshot(result) == annotations_before


def test_bipolar_qc_checks_channels_types_samples_sfreq_annotations_and_values() -> None:
    source, _ = make_four_channel_raw()
    result = apply_bipolar_reference(source, pairs())

    qc = validate_bipolar_output(source, result, pairs(), chunk_samples=333)

    assert qc.passed is True
    assert qc.errors == ()
    assert qc.channel_names == ("EEG_bipolar", "EMG_bipolar")
    assert qc.channel_types == ("eeg", "emg")
    assert qc.n_times == source.n_times
    assert qc.sfreq == source.info["sfreq"]
    assert qc.annotations_equal is True
    assert max(qc.max_abs_error.values()) < 1e-14


def test_wrong_bipolar_data_fails_numerical_qc() -> None:
    source, _ = make_four_channel_raw()
    result = apply_bipolar_reference(source, pairs())
    result._data[0, 100] += 1e-3

    qc = validate_bipolar_output(source, result, pairs(), chunk_samples=500)

    assert qc.passed is False
    assert "EEG_bipolar数值不等于正通道减参考通道" in qc.errors


def test_missing_source_channel_blocks_bipolar_reference() -> None:
    source, _ = make_four_channel_raw()
    source.drop_channels(["Channel4"])

    with pytest.raises(ValueError, match="缺少双极参考源通道.*Channel4"):
        apply_bipolar_reference(source, pairs())


def test_freeze_raw_reopens_and_saves_complete_evidence(tmp_path: Path) -> None:
    source, _ = make_four_channel_raw()
    bipolar = apply_bipolar_reference(source, pairs())
    destination = tmp_path / "双极参考后数据-20260727-1800-ver.001_raw.fif"
    evidence = tmp_path / "evidence"

    frozen = freeze_raw(
        bipolar,
        destination,
        evidence,
        now=datetime(2026, 7, 27, 18, 0),
    )

    assert frozen.path == destination.resolve()
    assert frozen.sha256 == sha256_file(destination)
    assert frozen.reopen_qc_passed is True
    assert frozen.channel_names == ("EEG_bipolar", "EMG_bipolar")
    assert frozen.channel_types == ("eeg", "emg")
    assert frozen.n_times == 2_000
    assert frozen.sfreq == 500.0
    assert len(frozen.evidence_paths) == 4
    assert all(path.exists() for path in frozen.evidence_paths)
    config_path = next(path for path in frozen.evidence_paths if "配置" in path.name)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["channel_names"] == ["EEG_bipolar", "EMG_bipolar"]
    assert config["sha256"] == frozen.sha256
    assert config["software_versions"]["mne"] == mne.__version__

    reopened = mne.io.read_raw_fif(destination, preload=True, verbose="ERROR")
    assert_allclose(reopened.get_data(), bipolar.get_data(), rtol=1e-6, atol=1e-12)
    assert annotation_snapshot(reopened) == annotation_snapshot(bipolar)


def test_freeze_raw_never_overwrites_existing_fif(tmp_path: Path) -> None:
    source, _ = make_four_channel_raw()
    bipolar = apply_bipolar_reference(source, pairs())
    destination = tmp_path / "双极参考后数据-20260727-1800-ver.001_raw.fif"
    freeze_raw(
        bipolar,
        destination,
        tmp_path / "first_evidence",
        now=datetime(2026, 7, 27, 18, 0),
    )
    original_hash = sha256_file(destination)

    with pytest.raises(FileExistsError):
        freeze_raw(
            bipolar,
            destination,
            tmp_path / "second_evidence",
            now=datetime(2026, 7, 27, 18, 1),
        )

    assert sha256_file(destination) == original_hash


def test_preartifact_checkpoint_is_byte_identical_non_overwriting_copy(
    tmp_path: Path,
) -> None:
    source, _ = make_four_channel_raw()
    bipolar = apply_bipolar_reference(source, pairs())
    first_path = tmp_path / "03" / "双极参考后数据-20260727-1800-ver.001_raw.fif"
    frozen = freeze_raw(
        bipolar,
        first_path,
        tmp_path / "evidence_03",
        now=datetime(2026, 7, 27, 18, 0),
    )
    preartifact_path = tmp_path / "04" / "去伪迹前数据-20260727-1801-ver.001_raw.fif"

    copied = copy_frozen_checkpoint(
        frozen,
        preartifact_path,
        tmp_path / "evidence_04",
        now=datetime(2026, 7, 27, 18, 1),
    )

    assert copied.sha256 == frozen.sha256
    assert copied.path.read_bytes() == frozen.path.read_bytes()
    assert copied.reopen_qc_passed is True
    with pytest.raises(FileExistsError):
        copy_frozen_checkpoint(
            frozen,
            preartifact_path,
            tmp_path / "evidence_again",
            now=datetime(2026, 7, 27, 18, 2),
        )
