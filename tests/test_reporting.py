from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.models import ChartAnalysis, ResearchQuestionConclusions
from eeg_pipeline.reporting import (
    ReportingInputs,
    RunEvidenceSummary,
    generate_result_charts,
    masked_relative_values,
    reporting_step_notices,
    write_completed_chart_reviews,
    write_final_result_report,
    _phase_display,
)


NOW = datetime(2026, 7, 27, 14, 5)


def _write_inputs(tmp_path: Path) -> ReportingInputs:
    bandpower_path = tmp_path / "bandpower.csv"
    rows = []
    phases = ("Baseline", "Treatment", "Recovery")
    bands = (
        "Delta",
        "Theta",
        "Sigma",
        "Beta",
        "Low_Gamma",
        "High_Gamma",
        "Alpha_additional",
    )
    for index in range(18):
        row = {
            "start_seconds": index * 2.5,
            "stop_seconds": index * 2.5 + 5.0,
            "midpoint_seconds": index * 2.5 + 2.5,
            "phase": phases[index // 6],
            "status": "Valid",
        }
        for band_index, band in enumerate(bands, start=1):
            row[f"{band}_relative"] = 0.01 * band_index + index * 0.001
        rows.append(row)
    pd.DataFrame(rows).to_csv(bandpower_path, index=False)

    epoch_path = tmp_path / "antila_epochs.csv"
    epoch_rows = []
    stages = (
        "Wake",
        "NREM",
        "REM",
        "Uncertain",
        "Artifact",
        "Boundary_Unscored",
    )
    for index in range(18):
        epoch_rows.append(
            {
                "start_seconds": index * 2.5,
                "stop_seconds": index * 2.5 + 5.0,
                "midpoint_seconds": index * 2.5 + 2.5,
                "phase": phases[index // 6],
                "sleep_stage": stages[index % len(stages)],
            }
        )
    pd.DataFrame(epoch_rows).to_csv(epoch_path, index=False)

    summary_path = tmp_path / "antila_summary.csv"
    summary_rows = []
    for phase in phases:
        for index, stage in enumerate(stages):
            summary_rows.append(
                {
                    "phase": phase,
                    "sleep_stage": stage,
                    "duration_seconds": 10 + index,
                    "percentage_of_phase": 10 + index,
                    "formal_sleep_duration": False,
                }
            )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return ReportingInputs(bandpower_path, epoch_path, summary_path)


def _analyses_from_manifest(manifest_path: Path) -> tuple[ChartAnalysis, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tuple(
        ChartAnalysis(
            chart_path=Path(item["chart_path"]),
            source_data_paths=tuple(Path(path) for path in item["source_data_paths"]),
            elements=item["elements"],
            observations="已逐图查看图像中的曲线、分布或时间轴。",
            numerical_evidence="已与同源CSV逐项核对中位数、比例或逐窗数值。",
            chart_conclusion="该图显示合成测试数据中的阶段差异。",
            n2o_relevance="测试数据不能用于判断N2O效应。",
            limitations="这是合成测试数据，不构成生物学证据。",
        )
        for item in manifest["charts"]
    )


def _evidence() -> RunEvidenceSummary:
    return RunEvidenceSummary(
        marker_summary="Marker由用户提供并已完成时间轴换算。",
        artifact_summary="伪迹决定及排除时长已记录。",
        line_noise_summary="50 Hz与100 Hz处理决定及前后PSD证据已记录。",
        reproducibility_summary="输入、冻结文件、软件版本和SHA256已记录。",
    )


def _conclusions() -> ResearchQuestionConclusions:
    return ResearchQuestionConclusions(
        acute_effect="本设计不可计算",
        acute_rationale="合成测试没有匹配Control。",
        recovery_0_2h_effect="本设计不可计算",
        recovery_0_2h_rationale="合成测试没有恢复对照。",
        long_term_persistence="不成立",
        long_term_rationale="合成测试没有长期持续变化。",
        sleep_structure_change="部分成立",
        sleep_structure_rationale="合成状态比例随阶段变化。",
        paper_parameter_consistency="本设计不可计算",
        paper_parameter_rationale="合成测试不对应论文实验。",
    )


def test_reporting_notices_define_six_numbered_steps() -> None:
    notices = reporting_step_notices()

    assert [notice.step_number for notice in notices] == [1, 2, 3, 4, 5, 6]
    assert all(notice.total_steps == 6 for notice in notices)
    assert "逐图" in notices[3].title
    assert "总报告" in notices[-1].title


def test_common_protocol_phase_names_are_displayed_in_chinese() -> None:
    assert _phase_display("Baseline") == "基线"
    assert _phase_display("Treatment") == "处理期"
    assert _phase_display("Recovery") == "恢复期"


def test_bandpower_time_series_keeps_invalid_windows_as_nan_gaps() -> None:
    frame = pd.DataFrame(
        {
            "status": ["Valid", "Artifact", "Boundary_Unscored", "Valid"],
            "Delta_relative": [0.1, 0.2, 0.3, 0.4],
        }
    )

    values = masked_relative_values(frame, "Delta_relative")

    assert values[0] == pytest.approx(0.1)
    assert np.isnan(values[1:3]).all()
    assert values[3] == pytest.approx(0.4)


def test_generate_result_charts_writes_seven_content_named_charts_and_pending_manifest(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)

    bundle = generate_result_charts(
        inputs,
        tmp_path / "charts",
        tmp_path / "reviews",
        now=NOW,
    )

    assert len(bundle.chart_paths) == 7
    assert all(path.exists() and path.stat().st_size > 0 for path in bundle.chart_paths)
    assert all("exploratory" not in path.name.lower() for path in bundle.chart_paths)
    assert all("current phase" not in path.name.lower() for path in bundle.chart_paths)
    manifest = json.loads(bundle.review_manifest_path.read_text(encoding="utf-8"))
    assert {item["review_status"] for item in manifest["charts"]} == {"Pending"}
    boxplots = [item for item in manifest["charts"] if item["chart_kind"] == "boxplot"]
    assert boxplots
    assert all("橙色线" in item["elements"] for item in boxplots)
    assert all("Q1" in item["elements"] and "Q3" in item["elements"] for item in boxplots)


def test_generate_result_charts_rejects_missing_required_bandpower_columns(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    frame = pd.read_csv(inputs.bandpower_csv_path)
    frame.drop(columns=["Delta_relative"]).to_csv(inputs.bandpower_csv_path, index=False)

    with pytest.raises(ValueError, match="Delta_relative"):
        generate_result_charts(inputs, tmp_path / "charts", tmp_path / "reviews", now=NOW)


def test_completed_chart_reviews_require_one_valid_analysis_per_chart(tmp_path: Path) -> None:
    bundle = generate_result_charts(
        _write_inputs(tmp_path), tmp_path / "charts", tmp_path / "reviews", now=NOW
    )
    analyses = _analyses_from_manifest(bundle.review_manifest_path)

    with pytest.raises(ValueError, match="全部图表"):
        write_completed_chart_reviews(
            bundle.review_manifest_path,
            analyses[:-1],
            tmp_path / "reviews",
            now=NOW,
        )


def test_final_report_contains_every_chart_explanation_and_separated_conclusion(
    tmp_path: Path,
) -> None:
    bundle = generate_result_charts(
        _write_inputs(tmp_path), tmp_path / "charts", tmp_path / "reviews", now=NOW
    )
    analyses = _analyses_from_manifest(bundle.review_manifest_path)
    completed = write_completed_chart_reviews(
        bundle.review_manifest_path,
        analyses,
        tmp_path / "reviews",
        now=NOW,
    )

    report_path = write_final_result_report(
        completed,
        analyses,
        _conclusions(),
        _evidence(),
        tmp_path / "reports",
        now=NOW,
    )

    html = report_path.read_text(encoding="utf-8")
    assert html.count("<img ") == len(bundle.chart_paths)
    assert "橙色线" in html
    assert "通气期间急性效应" in html
    assert "停止通气后0–2小时效应" in html
    assert "睡眠结构变化" in html
    assert "本设计不可计算" in html
    assert "需要更多动物" not in html


def test_final_report_rejects_invalid_research_question_verdict(tmp_path: Path) -> None:
    bundle = generate_result_charts(
        _write_inputs(tmp_path), tmp_path / "charts", tmp_path / "reviews", now=NOW
    )
    analyses = _analyses_from_manifest(bundle.review_manifest_path)
    completed = write_completed_chart_reviews(
        bundle.review_manifest_path,
        analyses,
        tmp_path / "reviews",
        now=NOW,
    )
    valid = _conclusions()
    invalid = ResearchQuestionConclusions(
        **{**valid.__dict__, "acute_effect": "支持"},  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="成立、部分成立、不成立、本设计不可计算"):
        write_final_result_report(
            completed,
            analyses,
            invalid,
            _evidence(),
            tmp_path / "reports",
            now=NOW,
        )
