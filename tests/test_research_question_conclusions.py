from __future__ import annotations

import pytest

from eeg_pipeline.models import ResearchQuestionConclusions
from eeg_pipeline.reporting import render_research_question_conclusions_html


def valid_conclusions() -> ResearchQuestionConclusions:
    return ResearchQuestionConclusions(
        acute_effect="成立",
        acute_rationale="σ波、β波和低γ波在当前比较中下降。",
        recovery_0_2h_effect="成立",
        recovery_0_2h_rationale="恢复0–2小时六频段均低于对照。",
        long_term_persistence="不成立",
        long_term_rationale="2小时后差异减弱并转为混合方向。",
        sleep_structure_change="部分成立",
        sleep_structure_rationale="状态构成改变，但方向不统一。",
        paper_parameter_consistency="部分成立",
        paper_parameter_rationale="急性σ/β/低γ同向，恢复指标不统一。",
    )


def test_research_question_conclusions_validate_and_render_directly() -> None:
    conclusions = valid_conclusions()
    assert conclusions.validate() == []
    html = render_research_question_conclusions_html(conclusions)
    for required in ("急性效应", "恢复0–2小时", "长期持续性", "睡眠结构", "成立", "不成立"):
        assert required in html
    for forbidden in ("需要更多动物", "有待大样本验证", "缺乏多只小鼠"):
        assert forbidden not in html


def test_no_control_can_be_reported_as_not_computable() -> None:
    conclusions = valid_conclusions()
    changed = ResearchQuestionConclusions(
        **{**conclusions.__dict__, "acute_effect": "本设计不可计算", "acute_rationale": "本记录没有匹配Control。"}
    )
    assert changed.validate() == []


@pytest.mark.parametrize("invalid", ["支持", "证据不足", "显著"])
def test_research_question_conclusions_reject_old_or_unknown_verdicts(invalid: str) -> None:
    conclusions = valid_conclusions()
    changed = ResearchQuestionConclusions(**{**conclusions.__dict__, "acute_effect": invalid})
    assert "成立、部分成立、不成立、本设计不可计算" in "；".join(changed.validate())

