from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.models import (  # noqa: E402
    BipolarPair,
    ChartAnalysis,
    MarkerSpec,
    ResearchQuestionConclusions,
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
    conclusion = ResearchQuestionConclusions(
        acute_effect="成立",
        acute_rationale="急性频段发生变化。",
        recovery_0_2h_effect="部分成立",
        recovery_0_2h_rationale="恢复期部分频段变化。",
        long_term_persistence="不成立",
        long_term_rationale="长期方向不一致。",
        sleep_structure_change="成立",
        sleep_structure_rationale="状态比例发生变化。",
        paper_parameter_consistency="部分成立",
        paper_parameter_rationale="部分参数与论文同向。",
    )

    assert config.validate() == []
    assert chart.validate() == []
    assert conclusion.validate() == []


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
