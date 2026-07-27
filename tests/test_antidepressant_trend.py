from eeg_pipeline.models import OverallConclusion
from eeg_pipeline.reporting import render_overall_conclusion_html


def test_single_animal_eeg_can_support_a_trend_without_claiming_efficacy() -> None:
    conclusion = OverallConclusion(
        electrophysiology_change="支持",
        paper_consistency="支持",
        antidepressant_like_trend="支持",
        antidepressant_effect="证据不足",
        trend_rationale="多个论文主频段在完整目标时段呈现与参考研究同向的变化。",
        rationale="这只是单只动物的电生理方向描述，尚未结合行为学终点。",
    )

    assert conclusion.validate(single_animal=True, eeg_only=True) == []


def test_antidepressant_like_trend_requires_an_explicit_rationale() -> None:
    conclusion = OverallConclusion(
        electrophysiology_change="证据不足",
        paper_consistency="证据不足",
        antidepressant_effect="证据不足",
        rationale="当前数据尚不能判断明确作用。",
    )

    assert "抗抑郁样趋势必须提供单独理由" in conclusion.validate(
        single_animal=True,
        eeg_only=True,
    )


def test_conclusion_output_separates_trend_from_efficacy_and_requires_behavior() -> None:
    conclusion = OverallConclusion(
        electrophysiology_change="支持",
        paper_consistency="支持",
        antidepressant_like_trend="支持",
        antidepressant_effect="证据不足",
        trend_rationale="完整记录中存在与论文方向一致的持续变化。",
        rationale="尚无同一只动物的行为学分析和群体重复。",
    )

    html = render_overall_conclusion_html(
        conclusion,
        single_animal=True,
        eeg_only=True,
    )

    assert "抗抑郁样趋势（单只动物EEG）" in html
    assert "完整记录" in html
    assert "更多动物" in html
    assert "行为学分析" in html
    assert "不能等同于已证实的抗抑郁疗效" in html
    assert "明显抗抑郁作用" in html
    assert "证据不足" in html


def test_skill_contract_requires_a_separate_antidepressant_like_trend_assessment() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    contract = (root / "references" / "chart-analysis-contract.md").read_text(
        encoding="utf-8"
    )
    combined = skill + contract

    for required in (
        "抗抑郁样趋势",
        "完整记录",
        "更多动物",
        "行为学分析",
        "不能等同",
    ):
        assert required in combined
