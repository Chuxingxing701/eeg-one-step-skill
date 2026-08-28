from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.antila_staging import (
    ANTILA_SLEEPY_SHA256,
    ANTILA_SOURCE_COMMIT,
    ANTILA_SOURCE_REPOSITORY,
    AntilaSourceSpec,
    AntilaFeatureArrays,
    compute_pooled_thresholds,
    align_antila_states,
    author_spectrum_window_count,
    validate_antila_source,
)


def test_antila_source_constants_and_author_window_count() -> None:
    assert ANTILA_SOURCE_REPOSITORY == "https://github.com/tortugar/Lab"
    assert ANTILA_SOURCE_COMMIT == "bcb8dae1594e64a511545e34f6050e2a417c1f45"
    assert ANTILA_SLEEPY_SHA256 == "da7ad9b5a771362cc529cca98a19879837337d4e40969f684067bcacc0148a04"
    assert author_spectrum_window_count(43_225_500, 500.0) == 34_581


def test_validate_antila_source_rejects_missing_checkout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Antila"):
        validate_antila_source(tmp_path / "missing")


def test_validate_antila_source_checks_commit_and_sleepy_hash(tmp_path: Path) -> None:
    root = tmp_path / "Lab"
    sleepy = root / "PySleep" / "sleepy.py"
    sleepy.parent.mkdir(parents=True)
    sleepy.write_text("# synthetic sleepy\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    import hashlib

    spec = AntilaSourceSpec(
        repository="fixture",
        commit=commit,
        sleepy_sha256=hashlib.sha256(sleepy.read_bytes()).hexdigest(),
    )
    assert validate_antila_source(root, spec=spec) == sleepy.resolve()
    wrong = AntilaSourceSpec("fixture", "0" * 40, spec.sleepy_sha256)
    with pytest.raises(RuntimeError, match="commit"):
        validate_antila_source(root, spec=wrong)


def test_align_antila_states_preserves_project_axis_and_invalid_labels() -> None:
    axis = pd.DataFrame(
        {
            "start_seconds": [0.0, 2.5, 5.0, 7.5],
            "stop_seconds": [5.0, 7.5, 10.0, 12.5],
            "midpoint_seconds": [2.5, 5.0, 7.5, 10.0],
            "phase": ["Pre", "Gas", "Gas", "Recovery"],
            "status": ["Valid", "Artifact", "Boundary_Unscored", "Valid"],
        }
    )
    aligned, tail = align_antila_states(axis, np.array([2, 3, 1, 0, 2, 2]))
    assert aligned["sleep_stage"].tolist() == ["Wake", "Artifact", "Boundary_Unscored", "Uncertain"]
    assert aligned["staging_method"].tolist() == ["Antila"] * 4
    assert tail["author_window_index"].tolist() == [4, 5]
    assert tail["sleep_stage"].tolist() == ["Wake", "Wake"]


def test_compute_pooled_thresholds_uses_only_valid_values() -> None:
    features = AntilaFeatureArrays(
        delta=np.array([1.0, 2.0, 100.0]),
        theta=np.array([2.0, 6.0, 100.0]),
        sigma=np.array([3.0, 5.0, 100.0]),
        gamma=np.array([4.0, 8.0, 100.0]),
        emg=np.array([10.0, 14.0, 100.0]),
    )
    thresholds = compute_pooled_thresholds(
        ((features, np.array([True, True, False])),)
    )
    assert thresholds.theta_delta == pytest.approx(3.0)
    assert thresholds.delta == pytest.approx(1.5)
    assert thresholds.sigma == pytest.approx(4.0)
    assert thresholds.gamma == pytest.approx(6.0)
    assert thresholds.emg == pytest.approx(12.0)
