from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.models import (  # noqa: E402
    BipolarPair,
    ChartAnalysis,
    MarkerSpec,
    OverallConclusion,
    RunConfig,
)


def valid_marker() -> MarkerSpec:
    return MarkerSpec(
        name="N2O_start",
        value="13560",
        basis="concatenated_seconds",
        source_file=None,
        meaning="开始吸入N2O",
    )


def valid_pairs() -> tuple[BipolarPair, ...]:
    return (
        BipolarPair(
            output_name="EEG_bipolar",
            positive_channel="Channel4",
            reference_channel="Channel3",
            channel_type="eeg",
        ),
        BipolarPair(
            output_name="EMG_bipolar",
            positive_channel="Channel2",
            reference_channel="Channel1",
            channel_type="emg",
        ),
    )


def valid_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        output_root=tmp_path / "results",
        input_files=(tmp_path / "recording.edf",),
        expected_sfreq=500.0,
        file_order_confirmed=True,
        same_animal_continuous_recording=True,
        bipolar_pairs=valid_pairs(),
        markers=(valid_marker(),),
    )


def test_valid_domain_models_have_no_validation_errors(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    chart = ChartAnalysis(
        chart_path=tmp_path / "Delta功率阶段箱线图.png",
        source_data_paths=(tmp_path / "bandpower.csv",),
        elements="橙色线为中位数，箱体为Q1至Q3。",
        observations="Treatment中位数高于Baseline。",
        numerical_evidence="Baseline中位数1.0，Treatment中位数1.2。",
        chart_conclusion="图中存在方向性差异。",
        n2o_relevance="支持进一步核查N2O相关变化。",
        limitations="单只动物，不能进行群体推断。",
    )
    conclusion = OverallConclusion(
        electrophysiology_change="支持",
        paper_consistency="证据不足",
        antidepressant_effect="证据不足",
        rationale="单只动物EEG只能描述个体内电生理变化。",
    )

    assert config.validate() == []
    assert chart.validate() == []
    assert conclusion.validate(single_animal=True, eeg_only=True) == []


def test_run_config_rejects_current_directory_as_output_root(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    invalid = RunConfig(
        output_root=Path(""),
        input_files=config.input_files,
        expected_sfreq=config.expected_sfreq,
        file_order_confirmed=config.file_order_confirmed,
        same_animal_continuous_recording=config.same_animal_continuous_recording,
        bipolar_pairs=config.bipolar_pairs,
        markers=config.markers,
    )

    assert "必须明确选择结果导出文件夹" in invalid.validate()


def test_run_config_rejects_duplicate_bipolar_output_names(tmp_path: Path) -> None:
    duplicate_pairs = (
        valid_pairs()[0],
        BipolarPair(
            output_name="EEG_bipolar",
            positive_channel="Channel2",
            reference_channel="Channel1",
            channel_type="emg",
        ),
    )
    config = valid_config(tmp_path)
    invalid = RunConfig(
        output_root=config.output_root,
        input_files=config.input_files,
        expected_sfreq=config.expected_sfreq,
        file_order_confirmed=config.file_order_confirmed,
        same_animal_continuous_recording=config.same_animal_continuous_recording,
        bipolar_pairs=duplicate_pairs,
        markers=config.markers,
    )

    assert "双极输出通道名不得重复" in invalid.validate()


def test_run_config_rejects_empty_marker_name(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    invalid_marker = MarkerSpec(
        name="  ",
        value="13560",
        basis="concatenated_seconds",
        source_file=None,
        meaning="开始吸入N2O",
    )
    invalid = RunConfig(
        output_root=config.output_root,
        input_files=config.input_files,
        expected_sfreq=config.expected_sfreq,
        file_order_confirmed=config.file_order_confirmed,
        same_animal_continuous_recording=config.same_animal_continuous_recording,
        bipolar_pairs=config.bipolar_pairs,
        markers=(invalid_marker,),
    )

    assert "Marker名称不能为空" in invalid.validate()


@pytest.mark.parametrize("invalid_verdict", ["maybe", "显著", "yes"])
def test_overall_conclusion_rejects_unknown_verdict(invalid_verdict: str) -> None:
    conclusion = OverallConclusion(
        electrophysiology_change=invalid_verdict,  # type: ignore[arg-type]
        paper_consistency="证据不足",
        antidepressant_effect="证据不足",
        rationale="测试非法结论值。",
    )

    assert "结论只能使用：支持、不支持、证据不足" in conclusion.validate(
        single_animal=True,
        eeg_only=True,
    )


def test_single_animal_eeg_only_cannot_claim_antidepressant_support() -> None:
    conclusion = OverallConclusion(
        electrophysiology_change="支持",
        paper_consistency="支持",
        antidepressant_effect="支持",
        rationale="错误地把电生理变化当作抗抑郁证据。",
    )

    assert "单只动物EEG不能支持明显抗抑郁作用" in conclusion.validate(
        single_animal=True,
        eeg_only=True,
    )
