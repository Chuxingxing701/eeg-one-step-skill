from pathlib import Path


SKILL_PATH = Path(__file__).parents[1] / "SKILL.md"


def test_skill_hilbert_gate_requires_smoke_before_full_candidate_run() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    gate = text[text.index("## Hilbert Candidate Staging Gate") :]
    positions = [gate.index(f"步骤 {index}/8") for index in range(1, 9)]
    assert positions == sorted(positions)
    assert gate.index("hilbert_method_explanation") < gate.index("Smoke Test")
    assert gate.index("Smoke Test") < gate.index("run_hilbert_candidate_staging")
    for required in (
        "用户提供的校准区间",
        "40个采样点",
        "不会去除心电",
        "Q25/Q50/Q75",
        "Wake_candidate",
        "NREM_candidate",
        "REM_candidate",
        "Uncertain",
        "Artifact",
        "Boundary_Unscored",
        "不是Ground Truth",
        "不删除短Bout",
        "人工复核",
    ):
        assert required in gate
