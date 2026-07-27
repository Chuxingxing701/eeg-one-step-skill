from datetime import datetime
import json
from pathlib import Path
import sys

import mne
import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.line_noise import (  # noqa: E402
    LineNoiseConfig,
    apply_line_noise_filter,
    compute_paper_psd,
    write_filter_evidence,
    write_psd_review_bundle,
)


def make_raw() -> mne.io.RawArray:
    sfreq = 500.0
    times = np.arange(int(15 * sfreq), dtype=float) / sfreq
    data = np.vstack(
        [
            30e-6 * np.sin(2 * np.pi * 10 * times)
            + 80e-6 * np.sin(2 * np.pi * 50 * times),
            20e-6 * np.sin(2 * np.pi * 25 * times)
            + 60e-6 * np.sin(2 * np.pi * 50 * times),
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(
            ["EEG_bipolar", "EMG_bipolar"],
            sfreq,
            ["eeg", "emg"],
        ),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 5.0],
            duration=[0.0, 1.0],
            description=["N2O_start", "BAD_auto_0001"],
        )
    )
    return raw


def test_filter_evidence_records_all_engineering_parameters_and_user_decision(
    tmp_path: Path,
) -> None:
    config = LineNoiseConfig()
    raw = make_raw()
    before = write_psd_review_bundle(
        compute_paper_psd(raw, config),
        tmp_path / "before",
        stage_label="滤波前",
        now=datetime(2026, 7, 27, 20, 30),
    )
    filtered = apply_line_noise_filter(raw, config, filter_100=True)
    after = write_psd_review_bundle(
        compute_paper_psd(filtered.raw, config),
        tmp_path / "after",
        stage_label="滤波后",
        now=datetime(2026, 7, 27, 20, 31),
    )

    config_path, annotations_path, report_path = write_filter_evidence(
        filtered,
        config,
        before,
        after,
        tmp_path / "evidence",
        now=datetime(2026, 7, 27, 20, 32),
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["filter_source"] == "工程处理，论文未报告陷波参数"
    assert payload["applied_frequencies_hz"] == [50.0, 100.0]
    assert payload["filter_100_user_decision"] is True
    assert payload["method"] == "fir"
    assert payload["phase"] == "zero"
    assert payload["fir_window"] == "hamming"
    assert payload["fir_design"] == "firwin"
    assert payload["filter_length"] == "auto"
    assert payload["trans_bandwidth_hz"] == 1.0
    assert payload["notch_widths_hz"] == [0.25, 0.5]
    assert payload["qc"]["annotations_equal"] is True
    assert payload["before_bundle"]["charts"] == [str(path) for path in before.chart_paths]
    assert payload["after_bundle"]["charts"] == [str(path) for path in after.chart_paths]

    annotations_text = annotations_path.read_text(encoding="utf-8-sig")
    assert "N2O_start" in annotations_text
    assert "BAD_auto_0001" in annotations_text
    report = report_path.read_text(encoding="utf-8")
    assert "论文未报告陷波参数" in report
    assert "100 Hz用户决定：滤除" in report
    assert "样本数、通道和Annotations结构QC：通过" in report
