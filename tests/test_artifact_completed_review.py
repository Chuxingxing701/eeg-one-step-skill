from datetime import datetime
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.artifacts import (  # noqa: E402
    ArtifactCandidate,
    ArtifactDetectionConfig,
    ArtifactDetectionResult,
    ArtifactReview,
    write_completed_artifact_reviews,
)


def detection() -> ArtifactDetectionResult:
    candidate = ArtifactCandidate(
        candidate_id="CANDIDATE_0001",
        onset_seconds=20.0,
        duration_seconds=5.0,
        possible_types=("运动或电缆扰动候选",),
        triggered_metrics=("EEG_bipolar rms稳健异常分数=9.1",),
        algorithm_reason="EEG与EMG同步偏离稳健基线，需要逐图复核。",
    )
    return ArtifactDetectionResult(
        config=ArtifactDetectionConfig(),
        candidates=(candidate,),
        window_count=10,
        skipped_existing_bad_count=0,
    )


def completed_review() -> ArtifactReview:
    return ArtifactReview(
        candidate_id="CANDIDATE_0001",
        verdict="支持伪迹",
        artifact_type="运动或电缆扰动",
        reason="两条通道同步出现异常宽幅紊乱，不符合邻近稳定波形。",
        image_evidence="红色候选区内EEG与EMG同时增宽，区间结束后恢复。",
    )


def test_completed_review_writes_one_reasoned_conclusion_beside_each_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "候选伪迹0001波形.png"
    image_path.write_bytes(b"synthetic-image-placeholder")

    json_path, markdown_path = write_completed_artifact_reviews(
        detection(),
        (completed_review(),),
        (image_path,),
        tmp_path,
        now=datetime(2026, 7, 27, 19, 0),
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    review = payload["candidates"][0]["codex_review"]
    assert review["status"] == "complete"
    assert review["verdict"] == "支持伪迹"
    assert review["artifact_type"] == "运动或电缆扰动"
    assert review["reason"]
    assert review["image_evidence"]

    text = markdown_path.read_text(encoding="utf-8")
    assert image_path.name in text
    for required in (
        "逐图复核结论：支持伪迹",
        "伪迹类型：运动或电缆扰动",
        completed_review().reason,
        completed_review().image_evidence,
    ):
        assert required in text


def test_completed_review_blocks_missing_image_or_incomplete_review(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="图片数量"):
        write_completed_artifact_reviews(
            detection(),
            (completed_review(),),
            (),
            tmp_path,
            now=datetime(2026, 7, 27, 19, 0),
        )

    with pytest.raises(ValueError, match="逐图复核未完成"):
        write_completed_artifact_reviews(
            detection(),
            (),
            (tmp_path / "missing.png",),
            tmp_path,
            now=datetime(2026, 7, 27, 19, 0),
        )
