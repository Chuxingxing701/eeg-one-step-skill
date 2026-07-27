from pathlib import Path


SKILL_PATH = Path(__file__).parents[1] / "SKILL.md"


def test_skill_line_noise_gate_announces_every_step_and_blocks_at_100hz() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    gate = text[text.index("## Line Noise Gate") :]

    positions = [gate.index(f"步骤 {index}/7") for index in range(1, 8)]
    assert positions == sorted(positions)
    assert gate.index("line_noise_method_explanation") < gate.index("compute_paper_psd")
    assert gate.index("compute_paper_psd") < gate.index("view_image")
    assert gate.index("view_image") < gate.index("filter_100")
    assert gate.index("filter_100") < gate.index("apply_line_noise_filter")
    for required in (
        "方法",
        "论文报告",
        "3秒Hann-Welch",
        "50%重叠",
        "工程处理",
        "50 Hz",
        "100 Hz",
        "用户明确决定",
        "样本数",
        "通道",
        "Annotations",
        "SHA256",
    ):
        assert required in gate
