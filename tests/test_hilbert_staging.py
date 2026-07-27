from datetime import datetime
import json
from pathlib import Path
import sys

import mne
from matplotlib import image as mpimg
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.bandpower import (  # noqa: E402
    BandpowerConfig,
    BandpowerResult,
    BandpowerRow,
    PhaseInterval,
    compute_bandpower,
    paper_band_definitions,
)
from eeg_pipeline.hilbert_staging import (  # noqa: E402
    CandidateThresholds,
    HilbertConfig,
    HilbertFeatureRow,
    SmokeWindow,
    build_candidate_bouts,
    classify_candidate_rows,
    hilbert_method_explanation,
    hilbert_step_notices,
    run_hilbert_candidate_staging,
    write_hilbert_evidence,
    write_hilbert_smoke_evidence,
)


def make_raw(duration_seconds: float = 20.0) -> mne.io.RawArray:
    sfreq = 500.0
    times = np.arange(int(duration_seconds * sfreq), dtype=float) / sfreq
    eeg = 30e-6 * np.sin(2 * np.pi * 3 * times)
    emg_amplitude = np.where(times < 10.0, 10e-6, 100e-6)
    emg = emg_amplitude * np.sin(2 * np.pi * 35 * times)
    return mne.io.RawArray(
        np.vstack([eeg, emg]),
        mne.create_info(
            ["EEG_bipolar", "EMG_bipolar"],
            sfreq,
            ["eeg", "emg"],
        ),
        verbose="ERROR",
    )


def make_bandpower() -> BandpowerResult:
    raw = make_raw()
    return compute_bandpower(
        raw,
        BandpowerConfig(channel_name="EEG_bipolar"),
        (
            PhaseInterval("Calibration", 0.0, 10.0),
            PhaseInterval("Treatment", 10.0, 20.0),
        ),
    )


def accepted_config() -> HilbertConfig:
    return HilbertConfig(
        exploratory_method_accepted=True,
        smoke_test_accepted=True,
    )


def test_hilbert_has_eight_prompts_and_explicit_limitations() -> None:
    notices = hilbert_step_notices(accepted_config())
    assert len(notices) == 8
    for index, notice in enumerate(notices, start=1):
        text = notice.render()
        assert f"[步骤 {index}/8]" in text
        assert "方法：" in text
        assert "目的：" in text
        assert "产出：" in text

    explanation = hilbert_method_explanation(accepted_config())
    for required in (
        "scipy.signal.hilbert",
        "40个采样点",
        "不会去除心电",
        "Q25/Q50/Q75",
        "Wake_candidate",
        "NREM_candidate",
        "REM_candidate",
        "Uncertain",
        "Artifact",
        "不是Ground Truth",
    ):
        assert required in explanation


def test_full_hilbert_run_requires_method_and_smoke_acceptance() -> None:
    raw = make_raw()
    bandpower = make_bandpower()
    calibration = PhaseInterval("Calibration", 0.0, 10.0)

    with pytest.raises(ValueError, match="探索性方法边界"):
        run_hilbert_candidate_staging(
            raw,
            bandpower,
            HilbertConfig(
                exploratory_method_accepted=False,
                smoke_test_accepted=True,
            ),
            calibration,
        )
    with pytest.raises(ValueError, match="Smoke Test"):
        run_hilbert_candidate_staging(
            raw,
            bandpower,
            HilbertConfig(
                exploratory_method_accepted=True,
                smoke_test_accepted=False,
            ),
            calibration,
        )


def test_smoke_test_writes_viewable_content_named_evidence(tmp_path: Path) -> None:
    bundle = write_hilbert_smoke_evidence(
        make_raw(),
        HilbertConfig(exploratory_method_accepted=True),
        (
            SmokeWindow("校准中段", 2.0, 4.0),
            SmokeWindow("处理段中段", 12.0, 14.0),
        ),
        tmp_path,
        now=datetime(2026, 7, 27, 21, 30),
    )

    assert bundle.summary_csv_path.exists()
    assert bundle.config_path.exists()
    assert bundle.report_path.exists()
    assert len(bundle.chart_paths) == 2
    for chart_path in bundle.chart_paths:
        assert chart_path.exists()
        assert chart_path.stat().st_size > 10_000
        assert "exploratory" not in chart_path.name.lower()
        assert "current phase" not in chart_path.name.lower()
        pixels = mpimg.imread(chart_path)
        assert float(np.std(pixels)) > 0.01
    payload = json.loads(bundle.config_path.read_text(encoding="utf-8"))
    assert payload["full_run_allowed"] is False
    assert payload["user_smoke_confirmation"] is None
    report = bundle.report_path.read_text(encoding="utf-8")
    assert "关闭图片后明确确认" in report
    assert "不会去除心电" in report


def thresholds() -> CandidateThresholds:
    return CandidateThresholds(
        calibration_name="Calibration",
        calibration_start_seconds=0.0,
        calibration_stop_seconds=10.0,
        valid_window_count=4,
        delta_q25=1.0,
        delta_q50=2.0,
        delta_q75=3.0,
        theta_delta_q25=1.0,
        theta_delta_q50=2.0,
        theta_delta_q75=3.0,
        sigma_q25=1.0,
        sigma_q50=2.0,
        sigma_q75=3.0,
        high_gamma_q25=1.0,
        high_gamma_q50=2.0,
        high_gamma_q75=3.0,
        hilbert_q25=1.0,
        hilbert_q50=2.0,
        hilbert_q75=3.0,
    )


def power_row(
    index: int,
    *,
    delta: float,
    theta: float,
    sigma: float,
    high_gamma: float,
    status: str = "Valid",
    phase: str = "Treatment",
) -> BandpowerRow:
    primary, additional = paper_band_definitions()
    names = [item.name for item in primary + additional]
    absolute = {name: 0.5 for name in names}
    absolute.update(
        {
            "Delta": delta,
            "Theta": theta,
            "Sigma": sigma,
            "High_Gamma": high_gamma,
        }
    )
    return BandpowerRow(
        start_seconds=index * 2.5,
        stop_seconds=index * 2.5 + 5.0,
        midpoint_seconds=index * 2.5 + 2.5,
        phase=phase,
        status=status,
        absolute_power=absolute,
        relative_power={name: 0.1 for name in names},
    )


def feature_row(index: int, median_uv: float) -> HilbertFeatureRow:
    return HilbertFeatureRow(
        start_seconds=index * 2.5,
        stop_seconds=index * 2.5 + 5.0,
        median_uv=median_uv,
        mean_uv=median_uv,
        p95_uv=median_uv,
        p99_uv=median_uv,
    )


def manual_bandpower(rows: tuple[BandpowerRow, ...]) -> BandpowerResult:
    primary, additional = paper_band_definitions()
    return BandpowerResult(
        rows=rows,
        primary_bands=primary,
        additional_bands=additional,
        window_seconds=5.0,
        step_seconds=2.5,
        welch_seconds=2.0,
        n_per_seg=1000,
        n_overlap=500,
        n_fft=1000,
    )


def test_candidate_rules_preserve_uncertain_artifact_and_boundary() -> None:
    rows = (
        power_row(0, delta=1.5, theta=1.5, sigma=1.0, high_gamma=1.0),
        power_row(1, delta=4.0, theta=6.0, sigma=1.0, high_gamma=1.0),
        power_row(2, delta=0.5, theta=2.0, sigma=1.0, high_gamma=1.0),
        power_row(3, delta=2.0, theta=4.0, sigma=2.0, high_gamma=2.0),
        power_row(
            4,
            delta=2.0,
            theta=2.0,
            sigma=2.0,
            high_gamma=2.0,
            status="Artifact",
        ),
        power_row(
            5,
            delta=2.0,
            theta=2.0,
            sigma=2.0,
            high_gamma=2.0,
            status="Boundary_Unscored",
            phase="Boundary",
        ),
    )
    features = tuple(
        feature_row(index, value)
        for index, value in enumerate((4.0, 2.0, 0.5, 2.0, 2.0, 2.0))
    )

    candidates = classify_candidate_rows(
        manual_bandpower(rows),
        features,
        thresholds(),
    )

    assert [row.candidate_stage for row in candidates] == [
        "Wake_candidate",
        "NREM_candidate",
        "REM_candidate",
        "Uncertain",
        "Artifact",
        "Boundary_Unscored",
    ]
    assert candidates[0].strict_rule_match_count == 1
    assert candidates[3].strict_rule_match_count == 0
    assert all(row.is_ground_truth is False for row in candidates)


def test_bouts_use_non_overlapping_midpoint_bins_without_short_bout_deletion() -> None:
    rows = (
        power_row(0, delta=1.5, theta=1.5, sigma=1.0, high_gamma=1.0),
        power_row(1, delta=1.5, theta=1.5, sigma=1.0, high_gamma=1.0),
        power_row(2, delta=4.0, theta=6.0, sigma=1.0, high_gamma=1.0),
    )
    features = (
        feature_row(0, 4.0),
        feature_row(1, 4.0),
        feature_row(2, 2.0),
    )
    candidates = classify_candidate_rows(
        manual_bandpower(rows),
        features,
        thresholds(),
    )

    bouts = build_candidate_bouts(candidates)

    assert len(bouts) == 2
    assert bouts[0].candidate_stage == "Wake_candidate"
    assert bouts[0].start_seconds == 0.0
    assert bouts[0].stop_seconds == 6.25
    assert bouts[1].candidate_stage == "NREM_candidate"
    assert bouts[1].start_seconds == 6.25
    assert bouts[1].stop_seconds == 10.0
    assert sum(item.duration_seconds for item in bouts) == 10.0


def test_hilbert_run_and_evidence_are_exploratory_and_non_overwriting(
    tmp_path: Path,
) -> None:
    result = run_hilbert_candidate_staging(
        make_raw(),
        make_bandpower(),
        accepted_config(),
        PhaseInterval("Calibration", 0.0, 10.0),
    )

    bundle = write_hilbert_evidence(
        result,
        tmp_path,
        now=datetime(2026, 7, 27, 22, 0),
    )

    assert bundle.epoch_csv_path.exists()
    assert bundle.bout_csv_path.exists()
    assert bundle.phase_summary_csv_path.exists()
    assert bundle.review_checklist_csv_path.exists()
    assert bundle.config_path.exists()
    assert bundle.report_path.exists()
    epoch_text = bundle.epoch_csv_path.read_text(encoding="utf-8-sig")
    assert "candidate_stage" in epoch_text
    assert "is_ground_truth" in epoch_text
    review_text = bundle.review_checklist_csv_path.read_text(encoding="utf-8-sig")
    for required in (
        "review_decision",
        "reviewer",
        "review_notes",
        "review_status",
        "suggested_review_start_seconds",
    ):
        assert required in review_text
    payload = json.loads(bundle.config_path.read_text(encoding="utf-8"))
    assert payload["formal_sleep_staging"] is False
    assert payload["fixed_wu_threshold_enabled"] is False
    assert payload["median_filter_samples"] == 40
    assert payload["manual_review_bouts_per_candidate_stage"] == 10
    assert payload["manual_review_context_seconds"] == 30.0
    report = bundle.report_path.read_text(encoding="utf-8")
    assert "不是Ground Truth" in report
    assert "Uncertain" in report
    assert "没有删除短Bout" in report
