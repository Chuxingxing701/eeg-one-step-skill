from pathlib import Path


SKILL_PATH = Path(__file__).parents[1] / "SKILL.md"


def test_skill_antila_gate_orders_source_calibration_run_and_alignment() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    gate = text[text.index("## Antila Staging Gate") : text.index("## Result Reporting Gate")]
    positions = [gate.index(f"步骤 {index}/8") for index in range(1, 9)]
    assert positions == sorted(positions)
    assert gate.index("validate_antila_source") < gate.index("use_idx")
    assert gate.index("use_idx") < gate.index("sleep_state")
    assert gate.index("sleep_state") < gate.index("align_antila_states")
    for required in (
        "tortugar/Lab",
        "bcb8dae1594e64a511545e34f6050e2a417c1f45",
        "pooled_valid",
        "Artifact",
        "Uncertain",
        "Boundary_Unscored",
        "零填充尾窗",
        "不得改写Bandpower值",
    ):
        assert required in gate

