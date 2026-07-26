"""Validate user-supplied EEG import facts before any file is opened."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, TypeVar
import json

from .models import BipolarPair, MarkerSpec, RunConfig
from .workspace import (
    WorkspacePaths,
    create_workspace,
    next_versioned_path,
    write_bytes_exclusive,
)


T = TypeVar("T")
WorkspaceFactory = Callable[[Path, str], WorkspacePaths]
Importer = Callable[[RunConfig], T]


@dataclass(frozen=True)
class ChecklistResult:
    valid: bool
    errors: tuple[str, ...]
    config: RunConfig | None
    payload: dict

    def require_valid(self) -> RunConfig:
        if not self.valid or self.config is None:
            message = "；".join(self.errors) or "Checklist未通过"
            raise ValueError(message)
        return self.config


def _positive_seconds(value: object, label: str, errors: list[str]) -> None:
    try:
        seconds = float(str(value))
    except (TypeError, ValueError):
        errors.append(f"{label}必须是非负秒数")
        return
    if seconds < 0:
        errors.append(f"{label}必须是非负秒数")


def _parse_expected_sfreq(value: object, errors: list[str]) -> float | None:
    if isinstance(value, str) and value.strip().lower() == "unknown":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append("预期采样率必须为正数或unknown")
        return None
    if result <= 0:
        errors.append("预期采样率必须为正数或unknown")
        return None
    return result


def _validate_marker_timing(
    marker: MarkerSpec,
    input_files: tuple[Path, ...],
    errors: list[str],
) -> None:
    if marker.basis in {"source_relative_seconds", "concatenated_seconds"}:
        _positive_seconds(marker.value, f"Marker {marker.name}时间", errors)

    known_sources = {
        candidate
        for path in input_files
        for candidate in (str(path), path.name)
    }
    if marker.basis == "source_relative_seconds":
        if marker.source_file and marker.source_file not in known_sources:
            errors.append(f"Marker所在文件不在输入清单中：{marker.source_file}")
    elif marker.basis == "concatenated_seconds":
        if marker.source_file and marker.source_file.strip():
            errors.append("合并后相对秒数不得再指定所在文件")
    elif marker.basis == "absolute_clock":
        if marker.source_file and marker.source_file.strip():
            errors.append("绝对时钟不得再指定所在文件")
        try:
            parsed = datetime.fromisoformat(marker.value.replace("Z", "+00:00"))
        except ValueError:
            errors.append("绝对时钟必须使用ISO 8601日期时间")
            return
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            errors.append("绝对时钟必须包含时区或UTC偏移")


def validate_checklist(payload: Mapping[str, object]) -> ChecklistResult:
    """Validate answers without checking whether EEG files exist or are readable."""

    saved_payload = deepcopy(dict(payload))
    errors: list[str] = []

    output_value = payload.get("output_root", "")
    output_root = Path(str(output_value))
    if str(output_root).strip() in {"", "."}:
        errors.append("必须明确选择结果导出文件夹")

    input_format = str(payload.get("input_format", "")).strip().lower()
    if input_format not in {"edf", "fif"}:
        errors.append("输入格式只能是edf或fif")

    raw_files = payload.get("input_files", [])
    input_files: tuple[Path, ...]
    if not isinstance(raw_files, list) or not raw_files:
        input_files = ()
        errors.append("至少需要一个EDF或FIF输入文件")
    else:
        input_files = tuple(Path(str(path)) for path in raw_files if str(path).strip())
        if len(input_files) != len(raw_files):
            errors.append("输入文件路径不能为空")
    if input_format == "fif" and len(input_files) > 1:
        errors.append("FIF输入模式只接受一个文件")
    if input_format in {"edf", "fif"}:
        expected_suffix = f".{input_format}"
        if any(path.suffix.lower() != expected_suffix for path in input_files):
            errors.append(f"输入文件后缀必须与{input_format}格式一致")

    expected_sfreq = _parse_expected_sfreq(
        payload.get("expected_sfreq"),
        errors,
    )
    file_order_confirmed = payload.get("file_order_confirmed") is True
    if not file_order_confirmed:
        errors.append("必须确认输入文件顺序")
    same_recording = payload.get("same_animal_continuous_recording") is True
    if not same_recording:
        errors.append("必须确认数据来自同一动物的连续记录")

    raw_channels = payload.get("channels", {})
    channels = raw_channels if isinstance(raw_channels, Mapping) else {}
    required_channel_roles = ("eeg", "eeg_ref", "emg", "emg_ref")
    for role in required_channel_roles:
        if not str(channels.get(role, "")).strip():
            errors.append(f"缺少通道身份：{role}")
    channel_values = [str(channels.get(role, "")).strip() for role in required_channel_roles]
    if all(channel_values) and len(channel_values) != len(set(channel_values)):
        errors.append("EEG/EMG及其参考通道必须是四个不同通道")

    raw_formulas = payload.get("bipolar_formulas", [])
    pairs: list[BipolarPair] = []
    if not isinstance(raw_formulas, list) or len(raw_formulas) != 2:
        errors.append("必须提供EEG和EMG两条双极公式")
    else:
        for raw_formula in raw_formulas:
            if not isinstance(raw_formula, Mapping):
                errors.append("双极公式格式无效")
                continue
            pairs.append(
                BipolarPair(
                    output_name=str(raw_formula.get("output_name", "")),
                    positive_channel=str(raw_formula.get("positive_channel", "")),
                    reference_channel=str(raw_formula.get("reference_channel", "")),
                    channel_type=str(raw_formula.get("channel_type", "")),  # type: ignore[arg-type]
                )
            )

    pair_by_type = {pair.channel_type: pair for pair in pairs}
    eeg_pair = pair_by_type.get("eeg")
    if eeg_pair and (
        eeg_pair.positive_channel != channels.get("eeg")
        or eeg_pair.reference_channel != channels.get("eeg_ref")
    ):
        errors.append("EEG双极公式必须与确认的EEG通道身份一致")
    emg_pair = pair_by_type.get("emg")
    if emg_pair and (
        emg_pair.positive_channel != channels.get("emg")
        or emg_pair.reference_channel != channels.get("emg_ref")
    ):
        errors.append("EMG双极公式必须与确认的EMG通道身份一致")
    if len(pair_by_type) != 2 or not {"eeg", "emg"}.issubset(pair_by_type):
        errors.append("双极公式必须各包含一条eeg和emg")

    raw_markers = payload.get("markers", [])
    markers: list[MarkerSpec] = []
    if not isinstance(raw_markers, list) or not raw_markers:
        errors.append("必须提供至少一个Marker")
    else:
        for raw_marker in raw_markers:
            if not isinstance(raw_marker, Mapping):
                errors.append("Marker格式无效")
                continue
            marker = MarkerSpec(
                name=str(raw_marker.get("name", "")),
                value=str(raw_marker.get("value", "")),
                basis=str(raw_marker.get("basis", "")),  # type: ignore[arg-type]
                source_file=(
                    str(raw_marker["source_file"])
                    if raw_marker.get("source_file") is not None
                    else None
                ),
                meaning=str(raw_marker.get("meaning", "")),
            )
            markers.append(marker)
            errors.extend(marker.validate())
            _validate_marker_timing(marker, input_files, errors)
    marker_names = [marker.name.strip() for marker in markers]
    if len(marker_names) != len(set(marker_names)):
        errors.append("Marker名称不得重复")

    if errors:
        return ChecklistResult(
            valid=False,
            errors=tuple(dict.fromkeys(errors)),
            config=None,
            payload=saved_payload,
        )

    config = RunConfig(
        output_root=output_root,
        input_files=input_files,
        expected_sfreq=expected_sfreq,
        file_order_confirmed=file_order_confirmed,
        same_animal_continuous_recording=same_recording,
        bipolar_pairs=tuple(pairs),
        markers=tuple(markers),
    )
    config_errors = config.validate()
    return ChecklistResult(
        valid=not config_errors,
        errors=tuple(config_errors),
        config=config if not config_errors else None,
        payload=saved_payload,
    )


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _serialize_config(config: RunConfig) -> dict:
    return {
        "output_root": str(config.output_root),
        "input_files": [str(path) for path in config.input_files],
        "expected_sfreq": config.expected_sfreq,
        "file_order_confirmed": config.file_order_confirmed,
        "same_animal_continuous_recording": config.same_animal_continuous_recording,
        "bipolar_pairs": [asdict(pair) for pair in config.bipolar_pairs],
        "markers": [asdict(marker) for marker in config.markers],
    }


def write_checklist_evidence(
    result: ChecklistResult,
    *,
    workspace: WorkspacePaths,
    run_id: str,
    confirmed_at: datetime,
) -> tuple[Path, ...]:
    """Write versioned JSON/Markdown evidence beneath the checklist step."""

    config = result.require_valid()
    if config.output_root.resolve() != workspace.output_root:
        raise ValueError("Checklist导出目录与当前工作区不一致")
    directory = workspace.directory("操作步骤/01_环境与Checklist")
    confirmed_text = confirmed_at.isoformat(timespec="seconds")

    checklist_json = next_versioned_path(
        directory,
        "导入Checklist",
        ".json",
        confirmed_at,
    )
    checklist_md = next_versioned_path(
        directory,
        "导入Checklist",
        ".md",
        confirmed_at,
    )
    config_json = next_versioned_path(
        directory,
        "运行配置",
        ".json",
        confirmed_at,
    )
    checklist_payload = {
        "run_id": run_id,
        "confirmed_at": confirmed_text,
        "valid": result.valid,
        "answers": result.payload,
    }
    markdown = (
        "# 导入前Checklist\n\n"
        f"- 运行编号：{run_id}\n"
        f"- 用户确认时间：{confirmed_text}\n"
        f"- 输入格式：{result.payload['input_format']}\n"
        f"- 输入文件数：{len(config.input_files)}\n"
        f"- Marker数量：{len(config.markers)}\n"
        f"- 双极公式数量：{len(config.bipolar_pairs)}\n"
        "- 检查结果：通过\n"
    )

    write_bytes_exclusive(checklist_json, _json_bytes(checklist_payload))
    write_bytes_exclusive(checklist_md, markdown.encode("utf-8"))
    write_bytes_exclusive(config_json, _json_bytes(_serialize_config(config)))
    return checklist_json, checklist_md, config_json


@dataclass
class ChecklistSession:
    """Small orchestration guard that enforces output-directory-first order."""

    run_id: str
    workspace_factory: WorkspaceFactory = create_workspace
    workspace: WorkspacePaths | None = None
    checklist_result: ChecklistResult | None = None

    def next_prompt(self) -> str:
        if self.workspace is None:
            return "请选择本次分析的结果导出文件夹。"
        if self.checklist_result is None:
            return "请完成导入前Checklist。"
        return "导入前Checklist已通过，可以进入文件读取。"

    def select_output_root(self, output_root: Path) -> WorkspacePaths:
        self.workspace = self.workspace_factory(output_root, self.run_id)
        return self.workspace

    def submit_checklist(self, payload: Mapping[str, object]) -> ChecklistResult:
        if self.workspace is None:
            raise RuntimeError("尚未选择结果导出文件夹")
        result = validate_checklist(payload)
        config = result.require_valid()
        if config.output_root.resolve() != self.workspace.output_root:
            raise ValueError("Checklist导出目录与已选择目录不一致")
        self.checklist_result = result
        return result

    def run_import(self, importer: Importer[T]) -> T:
        if self.workspace is None:
            raise RuntimeError("尚未选择结果导出文件夹")
        if self.checklist_result is None:
            raise RuntimeError("Checklist尚未通过")
        return importer(self.checklist_result.require_valid())
