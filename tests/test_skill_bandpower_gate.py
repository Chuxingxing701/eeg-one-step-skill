from pathlib import Path


SKILL_PATH = Path(__file__).parents[1] / "SKILL.md"


def test_skill_bandpower_gate_has_seven_steps_and_no_inferred_phases() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    gate = text[text.index("## Bandpower Gate") :]
    positions = [gate.index(f"步骤 {index}/7") for index in range(1, 8)]
    assert positions == sorted(positions)
    assert gate.index("bandpower_step_notices") < gate.index("compute_bandpower")
    for required in (
        "5秒窗",
        "2.5秒步长",
        "2秒Hann-Welch",
        "50%重叠",
        "Delta",
        "Theta",
        "Sigma",
        "Beta",
        "Low Gamma",
        "High Gamma",
        "Alpha仅为附加指标",
        "Artifact",
        "Boundary_Unscored",
        "NaN",
        "用户提供",
        "不得推断",
    ):
        assert required in gate
