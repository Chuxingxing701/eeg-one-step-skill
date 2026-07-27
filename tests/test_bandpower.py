from datetime import datetime
import json
from pathlib import Path
import sys

import mne
import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.bandpower import (  # noqa: E402
    BandpowerConfig,
    PhaseInterval,
    bandpower_step_notices,
    compute_bandpower,
    paper_band_definitions,
    write_bandpower_evidence,
)


def make_raw() -> mne.io.RawArray:
    sfreq = 500.0
    times = np.arange(int(30 * sfreq), dtype=float) / sfreq
    eeg = np.zeros(times.size)
    eeg[times < 10] = 50e-6 * np.sin(2 * np.pi * 2 * times[times < 10])
    middle = (times >= 10) & (times < 20)
    eeg[middle] = 50e-6 * np.sin(2 * np.pi * 7 * times[middle])
    late = times >= 20
    eeg[late] = 50e-6 * np.sin(2 * np.pi * 40 * times[late])
    raw = mne.io.RawArray(
        eeg[np.newaxis, :],
        mne.create_info(["EEG_bipolar"], sfreq, ["eeg"]),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[10.0, 20.0, 22.5],
            duration=[0.0, 0.0, 2.5],
            description=["N2O_start", "Recovery_start", "BAD_auto_0001"],
        )
    )
    return raw


def phases() -> tuple[PhaseInterval, ...]:
    return (
        PhaseInterval("Baseline", 0.0, 10.0),
        PhaseInterval("Treatment", 10.0, 20.0),
        PhaseInterval("Recovery", 20.0, 30.0),
    )


def config() -> BandpowerConfig:
    return BandpowerConfig(channel_name="EEG_bipolar")


def test_bandpower_has_seven_numbered_method_prompts() -> None:
    notices = bandpower_step_notices(config())
    assert len(notices) == 7
    for index, notice in enumerate(notices, start=1):
        text = notice.render()
        assert f"[步骤 {index}/7]" in text
        assert "方法：" in text
        assert "目的：" in text
        assert "产出：" in text


def test_paper_bands_and_additional_alpha_are_separate() -> None:
    primary, additional = paper_band_definitions()
    assert [(item.name, item.fmin, item.fmax) for item in primary] == [
        ("Delta", 0.5, 4.5),
        ("Theta", 6.0, 9.0),
        ("Sigma", 10.0, 15.0),
        ("Beta", 15.0, 30.0),
        ("Low_Gamma", 30.0, 50.0),
        ("High_Gamma", 50.0, 100.0),
    ]
    assert [(item.name, item.fmin, item.fmax) for item in additional] == [
        ("Alpha_additional", 8.0, 12.0),
    ]


def test_bandpower_includes_a_paper_band_upper_endpoint() -> None:
    sfreq = 500.0
    times = np.arange(int(10 * sfreq), dtype=float) / sfreq
    eeg = 50e-6 * np.sin(2 * np.pi * 4.5 * times)
    raw = mne.io.RawArray(
        eeg[np.newaxis, :],
        mne.create_info(["EEG_bipolar"], sfreq, ["eeg"]),
        verbose="ERROR",
    )

    result = compute_bandpower(
        raw,
        config(),
        (PhaseInterval("Baseline", 0.0, 10.0),),
    )

    assert result.rows[0].relative_power["Delta"] > 0.4


def test_bandpower_geometry_statuses_and_dominant_bands() -> None:
    result = compute_bandpower(make_raw(), config(), phases())

    assert result.window_seconds == 5.0
    assert result.step_seconds == 2.5
    assert result.welch_seconds == 2.0
    assert result.n_per_seg == 1000
    assert result.n_overlap == 500
    assert result.n_fft == 1000
    assert len(result.rows) == 11

    baseline = next(row for row in result.rows if row.start_seconds == 0.0)
    treatment = next(row for row in result.rows if row.start_seconds == 10.0)
    recovery = next(row for row in result.rows if row.start_seconds == 25.0)
    boundary = next(row for row in result.rows if row.start_seconds == 7.5)
    artifact = next(row for row in result.rows if row.start_seconds == 22.5)

    assert baseline.phase == "Baseline" and baseline.status == "Valid"
    assert treatment.phase == "Treatment" and treatment.status == "Valid"
    assert recovery.phase == "Recovery" and recovery.status == "Valid"
    assert baseline.absolute_power["Delta"] > baseline.absolute_power["Theta"] * 100
    assert treatment.absolute_power["Theta"] > treatment.absolute_power["Delta"] * 100
    assert recovery.absolute_power["Low_Gamma"] > recovery.absolute_power["Beta"] * 100
    assert boundary.status == "Boundary_Unscored"
    assert artifact.status == "Artifact"
    assert all(np.isnan(value) for value in boundary.absolute_power.values())
    assert all(np.isnan(value) for value in artifact.absolute_power.values())


def test_bandpower_evidence_writes_csv_config_and_method_report(tmp_path: Path) -> None:
    result = compute_bandpower(make_raw(), config(), phases())

    csv_path, config_path, report_path = write_bandpower_evidence(
        result,
        tmp_path,
        now=datetime(2026, 7, 27, 21, 0),
    )

    assert csv_path.exists() and config_path.exists() and report_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "Delta_absolute_V2" in csv_text
    assert "Theta_relative" in csv_text
    assert "Alpha_additional_absolute_V2" in csv_text
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["window_seconds"] == 5.0
    assert payload["step_seconds"] == 2.5
    assert payload["n_fft"] == 1000
    assert payload["invalid_value"] == "NaN"
    report = report_path.read_text(encoding="utf-8")
    assert "论文主频段" in report
    assert "Alpha仅为附加指标" in report
    assert "Artifact和Boundary_Unscored写为NaN，不写为0" in report
