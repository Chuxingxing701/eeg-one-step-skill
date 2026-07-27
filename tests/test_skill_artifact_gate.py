from pathlib import Path


SKILL_PATH = Path(__file__).parents[1] / "SKILL.md"


def test_skill_requires_method_preview_and_per_image_codex_review() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    method_position = text.index("artifact_method_explanation")
    detection_position = text.index("detect_artifact_candidates")
    image_review_position = text.index("view_image")
    user_gate_position = text.index("exclude_supported")

    assert method_position < detection_position < image_review_position < user_gate_position
    for required in (
        "常见伪迹",
        "每一张",
        "伪迹类型",
        "判断理由",
        "图像证据",
        "无法判断",
        "不得仅凭高EMG",
        "50 Hz",
        "100 Hz",
    ):
        assert required in text
