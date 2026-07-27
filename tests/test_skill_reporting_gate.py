from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL_PATH = ROOT / "SKILL.md"
CONTRACT_PATH = ROOT / "references" / "chart-analysis-contract.md"


def test_skill_reporting_gate_requires_image_and_source_review_before_report() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    gate = text[text.index("## Result Reporting Gate") :]
    positions = [gate.index(f"步骤 {index}/6") for index in range(1, 7)]
    assert positions == sorted(positions)
    assert gate.index("generate_result_charts") < gate.index("view_image")
    assert gate.index("view_image") < gate.index("write_completed_chart_reviews")
    assert gate.index("write_completed_chart_reviews") < gate.index("write_final_result_report")
    for required in (
        "橙色线",
        "Q1",
        "Q3",
        "1.5倍四分位距",
        "时频图",
        "同源CSV",
        "实验性结果",
        "需要人工核查",
        "电生理变化",
        "与论文方向一致性",
        "明显抗抑郁作用",
        "证据不足",
    ):
        assert required in gate


def test_chart_analysis_contract_exists_and_forbids_workflow_labels_in_chart_names() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    for required in (
        "exploratory",
        "current phase",
        "不得仅凭文件名",
        "逐张",
        "source_data_paths",
        "review_status",
    ):
        assert required in text

