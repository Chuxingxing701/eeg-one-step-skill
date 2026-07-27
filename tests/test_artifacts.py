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

from eeg_pipeline.artifacts import (  # noqa: E402
    ArtifactDetectionConfig,
    ArtifactReview,
    apply_artifact_batch_decision,
    artifact_method_explanation,
    attach_candidate_annotations,
    detect_artifact_candidates,
    validate_completed_reviews,
    write_artifact_review_bundle,
)


def make_bipolar_raw() -> mne.io.RawArray:
    sfreq = 100.0
    window_samples = int(5 * sfreq)
    window_count = 12
    rng = np.random.default_rng(20260727)
    eeg_windows = []
    emg_windows = []
    for index in range(window_count):
        scale = 1.0 + (index - window_count / 2) * 0.01
        eeg_windows.append(rng.normal(0.0, 20e-6 * scale, window_samples))
        emg_windows.append(rng.normal(0.0, 10e-6 * scale, window_samples))
    eeg = np.concatenate(eeg_windows)
    emg = np.concatenate(emg_windows)

    # Window 2 is already BAD and must not be reported again.
    eeg[2 * window_samples : 3 * window_samples] *= 100
    emg[2 * window_samples : 3 * window_samples] *= 100

    # Window 4 is a simultaneous high-amplitude disturbance.
    eeg[4 * window_samples : 5 * window_samples] *= 80
    emg[4 * window_samples : 5 * window_samples] *= 80

    # Window 7 is a complete signal dropout.
    eeg[7 * window_samples : 8 * window_samples] = 0.0
    emg[7 * window_samples : 8 * window_samples] = 0.0

    info = mne.create_info(
        ["EEG_bipolar", "EMG_bipolar"],
        sfreq=sfreq,
        ch_types=["eeg", "emg"],
    )
    raw = mne.io.RawArray(np.vstack([eeg, emg]), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 10.0],
            duration=[0.0, 5.0],
            description=["N2O_start", "BAD_existing"],
        )
    )
    return raw


def config() -> ArtifactDetectionConfig:
    return ArtifactDetectionConfig(
        window_seconds=5.0,
        robust_z_threshold=6.0,
        context_seconds=5.0,
        channel_names=("EEG_bipolar", "EMG_bipolar"),
    )


def annotation_snapshot(raw: mne.io.BaseRaw) -> tuple:
    return (
        tuple(raw.annotations.onset),
        tuple(raw.annotations.duration),
        tuple(raw.annotations.description),
        raw.annotations.orig_time,
    )


def reviews_for_detected_candidates(detection) -> tuple[ArtifactReview, ...]:
    first, second = detection.candidates
    return (
        ArtifactReview(
            candidate_id=first.candidate_id,
            verdict="支持伪迹",
            artifact_type="运动或电缆扰动",
            reason="EEG与EMG在同一窗口同步出现远高于邻近时段的宽幅紊乱。",
            image_evidence="两条波形同步增宽，候选区两侧迅速恢复到原有幅度。",
        ),
        ArtifactReview(
            candidate_id=second.candidate_id,
            verdict="不支持伪迹",
            artifact_type="未确认",
            reason="仅凭当前图像不能排除记录状态变化。",
            image_evidence="候选区幅度下降，但图像证据不足以确认设备中断。",
        ),
    )


def test_method_explanation_precedes_detection_and_teaches_common_artifacts() -> None:
    explanation = artifact_method_explanation(config())

    for required in (
        "5秒",
        "MAD",
        "6",
        "不会删除样本",
        "高幅瞬变",
        "运动或电缆扰动",
        "肌电污染",
        "平直线",
        "削顶或饱和",
        "50 Hz",
        "100 Hz",
        "高EMG不能单独证明伪迹",
    ):
        assert required in explanation


def test_detection_finds_extreme_and_flatline_but_skips_existing_bad() -> None:
    raw = make_bipolar_raw()
    data_before = raw.get_data().copy()
    annotations_before = annotation_snapshot(raw)

    detection = detect_artifact_candidates(raw, config())

    assert detection.window_count == 12
    assert detection.skipped_existing_bad_count == 1
    assert [item.onset_seconds for item in detection.candidates] == [20.0, 35.0]
    assert "运动或电缆扰动候选" in detection.candidates[0].possible_types
    assert "信号中断或平直线候选" in detection.candidates[1].possible_types
    assert all(item.algorithm_reason for item in detection.candidates)
    assert all(item.triggered_metrics for item in detection.candidates)
    assert_allclose(raw.get_data(), data_before, rtol=0.0, atol=0.0)
    assert annotation_snapshot(raw) == annotations_before


def test_candidate_annotations_are_non_bad_and_do_not_change_data() -> None:
    raw = make_bipolar_raw()
    detection = detect_artifact_candidates(raw, config())

    annotated = attach_candidate_annotations(raw, detection)

    assert annotated is not raw
    assert annotated.n_times == raw.n_times
    assert annotated.info["sfreq"] == raw.info["sfreq"]
    assert_allclose(annotated.get_data(), raw.get_data(), rtol=0.0, atol=0.0)
    candidate_descriptions = [
        value
        for value in annotated.annotations.description
        if value.startswith("CANDIDATE_")
    ]
    assert candidate_descriptions == [
        "CANDIDATE_0001",
        "CANDIDATE_0002",
    ]
    assert "N2O_start" in annotated.annotations.description
    assert "BAD_existing" in annotated.annotations.description


def test_review_bundle_has_one_nonblank_image_and_reason_per_candidate(
    tmp_path: Path,
) -> None:
    raw = make_bipolar_raw()
    detection = detect_artifact_candidates(raw, config())

    bundle = write_artifact_review_bundle(
        raw,
        detection,
        tmp_path,
        now=datetime(2026, 7, 27, 18, 30),
    )

    assert bundle.method_path.exists()
    assert bundle.candidates_csv_path.exists()
    assert bundle.review_manifest_path.exists()
    assert bundle.review_markdown_path.exists()
    assert len(bundle.image_paths) == len(detection.candidates)
    assert all(path.exists() and path.stat().st_size > 1_000 for path in bundle.image_paths)
    assert all(float(np.std(mpimg.imread(path))) > 0.0 for path in bundle.image_paths)
    assert all("exploratory" not in path.name.lower() for path in bundle.image_paths)
    assert all("current phase" not in path.name.lower() for path in bundle.image_paths)

    manifest = json.loads(bundle.review_manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["candidates"]) == len(detection.candidates)
    for item in manifest["candidates"]:
        assert item["algorithm_reason"]
        assert item["possible_types"]
        assert item["image_path"]
        assert item["codex_review"] == {
            "status": "pending",
            "verdict": "",
            "artifact_type": "",
            "reason": "",
            "image_evidence": "",
        }

    review_text = bundle.review_markdown_path.read_text(encoding="utf-8")
    for candidate in detection.candidates:
        assert candidate.candidate_id in review_text
        assert candidate.algorithm_reason in review_text
    for required in (
        "算法入选理由",
        "可能伪迹类型",
        "Codex逐图复核结论",
        "判断伪迹的理由",
        "图像证据",
    ):
        assert required in review_text


def test_review_validation_requires_every_candidate_and_every_reason() -> None:
    raw = make_bipolar_raw()
    detection = detect_artifact_candidates(raw, config())
    incomplete = (
        ArtifactReview(
            candidate_id=detection.candidates[0].candidate_id,
            verdict="支持伪迹",
            artifact_type="",
            reason="",
            image_evidence="",
        ),
    )

    errors = validate_completed_reviews(detection, incomplete)

    assert any("缺少逐图复核" in error for error in errors)
    assert any("伪迹类型不能为空" in error for error in errors)
    assert any("判断理由不能为空" in error for error in errors)
    assert any("图像证据不能为空" in error for error in errors)


def test_user_batch_decision_marks_only_codex_supported_candidates_bad() -> None:
    raw = make_bipolar_raw()
    detection = detect_artifact_candidates(raw, config())
    candidate_raw = attach_candidate_annotations(raw, detection)
    reviews = reviews_for_detected_candidates(detection)

    result = apply_artifact_batch_decision(
        candidate_raw,
        detection,
        reviews,
        decision="exclude_supported",
    )

    assert result.decision == "exclude_supported"
    assert result.bad_count == 1
    assert result.bad_duration_seconds == 5.0
    assert result.raw.n_times == raw.n_times
    assert result.raw.info["sfreq"] == raw.info["sfreq"]
    assert_allclose(result.raw.get_data(), raw.get_data(), rtol=0.0, atol=0.0)
    assert "BAD_auto_0001" in result.raw.annotations.description
    assert "BAD_auto_0002" not in result.raw.annotations.description
    assert "N2O_start" in result.raw.annotations.description


def test_batch_decision_blocks_pending_review_and_can_retain_all() -> None:
    raw = make_bipolar_raw()
    detection = detect_artifact_candidates(raw, config())
    candidate_raw = attach_candidate_annotations(raw, detection)

    with pytest.raises(ValueError, match="逐图复核未完成"):
        apply_artifact_batch_decision(
            candidate_raw,
            detection,
            (),
            decision="exclude_supported",
        )

    retained = apply_artifact_batch_decision(
        candidate_raw,
        detection,
        reviews_for_detected_candidates(detection),
        decision="retain_all",
    )
    assert retained.bad_count == 0
    assert not any(
        value.startswith("BAD_auto_")
        for value in retained.raw.annotations.description
    )
