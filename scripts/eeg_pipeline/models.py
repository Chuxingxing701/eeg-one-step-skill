"""Stable, side-effect-free domain models for the EEG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


MarkerBasis = Literal[
    "source_relative_seconds",
    "concatenated_seconds",
    "absolute_clock",
]
GateStatus = Literal["pending", "waiting_user", "complete", "failed"]
Decision = Literal["mark_all_bad", "retain_all"]
Verdict = Literal["支持", "不支持", "证据不足"]

VALID_MARKER_BASES = {
    "source_relative_seconds",
    "concatenated_seconds",
    "absolute_clock",
}
VALID_VERDICTS = {"支持", "不支持", "证据不足"}


@dataclass(frozen=True)
class MarkerSpec:
    """A user-supplied experimental marker that must never be inferred."""

    name: str
    value: str
    basis: MarkerBasis
    source_file: str | None
    meaning: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name.strip():
            errors.append("Marker名称不能为空")
        if not self.value.strip():
            errors.append("Marker时间不能为空")
        if self.basis not in VALID_MARKER_BASES:
            errors.append("Marker计时方式无效")
        if self.basis == "source_relative_seconds" and not (
            self.source_file and self.source_file.strip()
        ):
            errors.append("源文件内相对秒数必须提供所在文件")
        if not self.meaning.strip():
            errors.append("Marker阶段含义不能为空")
        return errors


@dataclass(frozen=True)
class BipolarPair:
    """One user-confirmed positive-minus-reference channel formula."""

    output_name: str
    positive_channel: str
    reference_channel: str
    channel_type: Literal["eeg", "emg"]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.output_name.strip():
            errors.append("双极输出通道名不能为空")
        if not self.positive_channel.strip():
            errors.append("双极正通道不能为空")
        if not self.reference_channel.strip():
            errors.append("双极参考通道不能为空")
        if self.positive_channel.strip() == self.reference_channel.strip():
            errors.append("双极正通道与参考通道不能相同")
        if self.channel_type not in {"eeg", "emg"}:
            errors.append("双极通道类型只能是eeg或emg")
        return errors


@dataclass(frozen=True)
class RunConfig:
    """User-approved inputs required before any EEG file is opened."""

    output_root: Path
    input_files: tuple[Path, ...]
    expected_sfreq: float | None
    file_order_confirmed: bool
    same_animal_continuous_recording: bool
    bipolar_pairs: tuple[BipolarPair, ...]
    markers: tuple[MarkerSpec, ...]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if str(self.output_root).strip() in {"", "."}:
            errors.append("必须明确选择结果导出文件夹")
        if not self.input_files:
            errors.append("至少需要一个EDF或FIF输入文件")
        if not self.file_order_confirmed:
            errors.append("必须确认输入文件顺序")
        if not self.same_animal_continuous_recording:
            errors.append("必须确认数据来自同一动物的连续记录")
        if self.expected_sfreq is not None and self.expected_sfreq <= 0:
            errors.append("预期采样率必须为正数或未知")
        if not self.bipolar_pairs:
            errors.append("必须提供双极参考公式")
        if not self.markers:
            errors.append("必须提供至少一个Marker")

        output_names = [pair.output_name.strip() for pair in self.bipolar_pairs]
        if len(output_names) != len(set(output_names)):
            errors.append("双极输出通道名不得重复")

        for pair in self.bipolar_pairs:
            errors.extend(pair.validate())
        for marker in self.markers:
            errors.extend(marker.validate())
        return list(dict.fromkeys(errors))


@dataclass(frozen=True)
class ChartAnalysis:
    """Evidence-backed interpretation that must accompany one chart."""

    chart_path: Path
    source_data_paths: tuple[Path, ...]
    elements: str
    observations: str
    numerical_evidence: str
    chart_conclusion: str
    n2o_relevance: str
    limitations: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if str(self.chart_path).strip() in {"", "."}:
            errors.append("图表路径不能为空")
        if not self.source_data_paths:
            errors.append("每张图必须声明同源数据文件")
        text_fields = {
            "图表元素解释": self.elements,
            "具体观察": self.observations,
            "数值证据": self.numerical_evidence,
            "图级结论": self.chart_conclusion,
            "N2O关系": self.n2o_relevance,
            "局限": self.limitations,
        }
        for label, value in text_fields.items():
            if not value.strip():
                errors.append(f"{label}不能为空")
        return errors


@dataclass(frozen=True)
class OverallConclusion:
    """Separate direction, trend, and efficacy verdicts for the final report."""

    electrophysiology_change: Verdict
    paper_consistency: Verdict
    antidepressant_effect: Verdict
    rationale: str
    antidepressant_like_trend: Verdict = "证据不足"
    trend_rationale: str = ""

    def validate(
        self,
        *,
        single_animal: bool,
        eeg_only: bool,
    ) -> list[str]:
        errors: list[str] = []
        verdicts = (
            cast(str, self.electrophysiology_change),
            cast(str, self.paper_consistency),
            cast(str, self.antidepressant_like_trend),
            cast(str, self.antidepressant_effect),
        )
        if any(value not in VALID_VERDICTS for value in verdicts):
            errors.append("结论只能使用：支持、不支持、证据不足")
        if (
            single_animal
            and eeg_only
            and self.antidepressant_effect == "支持"
        ):
            errors.append("单只动物EEG不能支持明显抗抑郁作用")
        if not self.rationale.strip():
            errors.append("总体结论必须提供理由")
        if not self.trend_rationale.strip():
            errors.append("抗抑郁样趋势必须提供单独理由")
        return errors
