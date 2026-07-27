from pathlib import Path


def test_skill_requires_documented_analysis_channel_selection() -> None:
    skill = (Path(__file__).parents[1] / "SKILL.md").read_text(encoding="utf-8")

    for required in (
        "双极公式涉及的用户确认分析通道",
        "排除通道",
        "EEG/EMG类型",
        "原始EDF",
        "不得修改",
    ):
        assert required in skill
