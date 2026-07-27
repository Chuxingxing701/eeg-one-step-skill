from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib.image as mpimg
import mne
import numpy as np
from numpy.testing import assert_allclose
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.line_noise import (  # noqa: E402
    LineNoiseConfig,
    apply_line_noise_filter,
    compute_paper_psd,
    line_noise_method_explanation,
    line_noise_step_notices,
    measure_line_noise,
    validate_line_noise_input,
    write_psd_review_bundle,
)


def make_line_noise_raw(duration_seconds: float = 30.0) -> mne.io.RawArray:
    sfreq = 500.0
    times = np.arange(int(duration_seconds * sfreq), dtype=float) / sfreq
    rng = np.random.default_rng(20260727)
    eeg = (
        25e-6 * np.sin(2 * np.pi * 10 * times)
        + 120e-6 * np.sin(2 * np.pi * 50 * times)
        + 80e-6 * np.sin(2 * np.pi * 100 * times)
        + rng.normal(0.0, 2e-6, times.size)
    )
    emg = (
        20e-6 * np.sin(2 * np.pi * 25 * times)
        + 90e-6 * np.sin(2 * np.pi * 50 * times)
        + 60e-6 * np.sin(2 * np.pi * 100 * times)
        + rng.normal(0.0, 3e-6, times.size)
    )
    info = mne.create_info(
        ["EEG_bipolar", "EMG_bipolar"],
        sfreq=sfreq,
        ch_types=["eeg", "emg"],
    )
    raw = mne.io.RawArray(np.vstack([eeg, emg]), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 10.0],
            duration=[0.0, 3.0],
            description=["N2O_start", "BAD_auto_0001"],
        )
    )
    return raw


def config() -> LineNoiseConfig:
    return LineNoiseConfig(
        channel_names=("EEG_bipolar", "EMG_bipolar"),
        psd_window_seconds=3.0,
        overlap_fraction=0.5,
        fmin=0.5,
        fmax=120.0,
        required_frequency_hz=50.0,
        optional_frequency_hz=100.0,
    )


def annotation_snapshot(raw: mne.io.BaseRaw) -> tuple:
    return (
        tuple(raw.annotations.onset),
        tuple(raw.annotations.duration),
        tuple(raw.annotations.description),
        raw.annotations.orig_time,
    )


def evidence_by_key(evidence):
    return {
        (item.channel_name, item.target_frequency_hz): item
        for item in evidence
    }


def test_every_line_noise_action_has_numbered_method_prompt() -> None:
    notices = line_noise_step_notices(config())

    assert len(notices) == 7
    assert [notice.step_number for notice in notices] == list(range(1, 8))
    for index, notice in enumerate(notices, start=1):
        rendered = notice.render()
        assert f"[步骤 {index}/7]" in rendered
        assert "方法：" in rendered
        assert "目的：" in rendered
        assert "产出：" in rendered
    assert "论文" in notices[1].method
    assert "工程" in notices[4].method
    assert "用户" in notices[3].method


def test_method_explanation_separates_paper_psd_from_engineering_notch() -> None:
    explanation = line_noise_method_explanation(config())

    for required in (
        "论文报告",
        "3秒",
        "Hann",
        "50%",
        "未报告陷波",
        "工程处理",
        "50 Hz",
        "100 Hz",
        "零相位FIR",
        "不执行0.5-100 Hz带通",
    ):
        assert required in explanation


def test_paper_psd_includes_eeg_and_emg_with_exact_welch_geometry() -> None:
    raw = make_line_noise_raw()
    qc = validate_line_noise_input(raw, config())

    psd = compute_paper_psd(raw, config())

    assert qc.passed is True
    assert psd.channel_names == ("EEG_bipolar", "EMG_bipolar")
    assert psd.values.shape[0] == 2
    assert psd.n_per_seg == 1500
    assert psd.n_overlap == 750
    assert psd.n_fft == 1500
    assert np.isclose(np.diff(psd.frequencies_hz).mean(), 1 / 3)
    assert psd.frequencies_hz[0] >= 0.5
    assert psd.frequencies_hz[-1] <= 120.0
    assert np.all(np.isfinite(psd.values))


def test_line_noise_evidence_reports_ratios_without_automatic_verdict() -> None:
    psd = compute_paper_psd(make_line_noise_raw(), config())

    evidence = measure_line_noise(psd, (50.0, 100.0))
    by_key = evidence_by_key(evidence)

    assert len(evidence) == 4
    for channel_name in psd.channel_names:
        for target in (50.0, 100.0):
            item = by_key[(channel_name, target)]
            assert abs(item.nearest_bin_hz - target) < 1e-9
            assert item.peak_to_local_ratio > 10.0
            assert item.is_local_maximum is True
            assert not hasattr(item, "automatic_verdict")


def test_prefilter_bundle_writes_three_named_nonblank_charts_and_sources(
    tmp_path: Path,
) -> None:
    psd = compute_paper_psd(make_line_noise_raw(), config())

    bundle = write_psd_review_bundle(
        psd,
        tmp_path,
        stage_label="滤波前",
        now=datetime(2026, 7, 27, 20, 0),
    )

    assert bundle.psd_csv_path.exists()
    assert bundle.evidence_csv_path.exists()
    assert bundle.config_path.exists()
    assert bundle.report_path.exists()
    assert bundle.review_manifest_path.exists()
    assert len(bundle.chart_paths) == 3
    assert any("完整PSD" in path.name for path in bundle.chart_paths)
    assert any("45至55HzPSD" in path.name for path in bundle.chart_paths)
    assert any("95至105HzPSD" in path.name for path in bundle.chart_paths)
    assert all(path.stat().st_size > 1_000 for path in bundle.chart_paths)
    assert all(float(np.std(mpimg.imread(path))) > 0.0 for path in bundle.chart_paths)
    assert all("exploratory" not in path.name.lower() for path in bundle.chart_paths)

    manifest = json.loads(bundle.review_manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["charts"]) == 3
    assert all(item["meaning"] for item in manifest["charts"])
    assert all(item["look_for"] for item in manifest["charts"])
    assert all(item["codex_review"]["status"] == "pending" for item in manifest["charts"])
    report = bundle.report_path.read_text(encoding="utf-8")
    assert "100 Hz是否滤除必须由用户决定" in report
    assert "窄而尖的局部峰" in report


def test_filter_blocks_missing_100hz_decision_and_preserves_structure() -> None:
    raw = make_line_noise_raw()
    data_before = raw.get_data().copy()
    annotations_before = annotation_snapshot(raw)

    with pytest.raises(ValueError, match="100 Hz尚未由用户决定"):
        apply_line_noise_filter(raw, config(), filter_100=None)

    result = apply_line_noise_filter(raw, config(), filter_100=False)

    assert result.frequencies_hz == (50.0,)
    assert result.qc.passed is True
    assert result.raw is not raw
    assert result.raw.ch_names == raw.ch_names
    assert result.raw.get_channel_types() == raw.get_channel_types()
    assert result.raw.n_times == raw.n_times
    assert result.raw.info["sfreq"] == raw.info["sfreq"]
    assert annotation_snapshot(result.raw) == annotations_before
    assert_allclose(raw.get_data(), data_before, rtol=0.0, atol=0.0)
    assert not np.allclose(result.raw.get_data(), data_before, rtol=0.0, atol=1e-15)


def test_50_and_user_approved_100hz_notches_reduce_their_psd_peaks() -> None:
    raw = make_line_noise_raw()
    before = compute_paper_psd(raw, config())
    filtered = apply_line_noise_filter(raw, config(), filter_100=True)
    after = compute_paper_psd(filtered.raw, config())
    before_evidence = evidence_by_key(measure_line_noise(before, (50.0, 100.0)))
    after_evidence = evidence_by_key(measure_line_noise(after, (50.0, 100.0)))

    assert filtered.frequencies_hz == (50.0, 100.0)
    for key, before_item in before_evidence.items():
        after_item = after_evidence[key]
        assert after_item.peak_psd < before_item.peak_psd * 0.1
