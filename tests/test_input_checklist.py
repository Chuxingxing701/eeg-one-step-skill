from datetime import datetime
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.input_checklist import (  # noqa: E402
    ChecklistSession,
    validate_checklist,
    write_checklist_evidence,
)
from eeg_pipeline.workspace import create_workspace  # noqa: E402


def valid_payload(tmp_path: Path) -> dict:
    file_6 = tmp_path / "mouse_6.edf"
    file_7 = tmp_path / "mouse_7.edf"
    return {
        "output_root": str(tmp_path / "results"),
        "input_format": "edf",
        "input_files": [str(file_6), str(file_7)],
        "expected_sfreq": 500.0,
        "file_order_confirmed": True,
        "same_animal_continuous_recording": True,
        "channels": {
            "eeg": "Channel4",
            "eeg_ref": "Channel3",
            "emg": "Channel2",
            "emg_ref": "Channel1",
        },
        "bipolar_formulas": [
            {
                "output_name": "EEG_bipolar",
                "positive_channel": "Channel4",
                "reference_channel": "Channel3",
                "channel_type": "eeg",
            },
            {
                "output_name": "EMG_bipolar",
                "positive_channel": "Channel2",
                "reference_channel": "Channel1",
                "channel_type": "emg",
            },
        ],
        "markers": [
            {
                "name": "Box_start",
                "value": "12600",
                "basis": "source_relative_seconds",
                "source_file": "mouse_6.edf",
                "meaning": "Box适应期开始",
            },
            {
                "name": "N2O_start",
                "value": "13560",
                "basis": "source_relative_seconds",
                "source_file": "mouse_6.edf",
                "meaning": "开始吸入N2O",
            },
            {
                "name": "Recovery_start",
                "value": "3600",
                "basis": "source_relative_seconds",
                "source_file": "mouse_7.edf",
                "meaning": "结束通气并进入恢复期",
            },
        ],
    }


def test_valid_checklist_builds_run_config_without_reading_files(
    tmp_path: Path,
) -> None:
    result = validate_checklist(valid_payload(tmp_path))

    assert result.valid is True
    assert result.errors == ()
    assert result.config is not None
    assert result.config.expected_sfreq == 500.0
    assert [marker.name for marker in result.config.markers] == [
        "Box_start",
        "N2O_start",
        "Recovery_start",
    ]
    assert result.config.bipolar_pairs[0].positive_channel == "Channel4"
    assert result.config.bipolar_pairs[0].reference_channel == "Channel3"


def test_unknown_sampling_rate_becomes_none(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["expected_sfreq"] = "unknown"

    result = validate_checklist(payload)

    assert result.valid is True
    assert result.config is not None
    assert result.config.expected_sfreq is None


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (lambda payload: payload.pop("input_files"), "至少需要一个EDF或FIF输入文件"),
        (lambda payload: payload["channels"].pop("eeg"), "缺少通道身份：eeg"),
        (lambda payload: payload.update(markers=[]), "必须提供至少一个Marker"),
        (
            lambda payload: payload.update(file_order_confirmed=False),
            "必须确认输入文件顺序",
        ),
        (
            lambda payload: payload.update(same_animal_continuous_recording=False),
            "必须确认数据来自同一动物的连续记录",
        ),
    ],
)
def test_required_checklist_items_block_progress(
    tmp_path: Path,
    mutator,
    expected_error: str,
) -> None:
    payload = valid_payload(tmp_path)
    mutator(payload)

    result = validate_checklist(payload)

    assert result.valid is False
    assert expected_error in result.errors
    assert result.config is None


def test_source_relative_marker_requires_a_known_source_file(
    tmp_path: Path,
) -> None:
    payload = valid_payload(tmp_path)
    payload["markers"][0]["source_file"] = "unknown.edf"

    result = validate_checklist(payload)

    assert "Marker所在文件不在输入清单中：unknown.edf" in result.errors


def test_concatenated_seconds_rejects_source_file_conflict(
    tmp_path: Path,
) -> None:
    payload = valid_payload(tmp_path)
    payload["markers"][0].update(
        basis="concatenated_seconds",
        source_file="mouse_6.edf",
    )

    result = validate_checklist(payload)

    assert "合并后相对秒数不得再指定所在文件" in result.errors


def test_absolute_clock_requires_timezone(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["markers"][0].update(
        basis="absolute_clock",
        value="2026-01-26T11:29:59",
        source_file=None,
    )

    result = validate_checklist(payload)

    assert "绝对时钟必须包含时区或UTC偏移" in result.errors


def test_bipolar_formula_must_match_channel_identity(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["bipolar_formulas"][0]["positive_channel"] = "Channel3"

    result = validate_checklist(payload)

    assert "EEG双极公式必须与确认的EEG通道身份一致" in result.errors


def test_output_directory_is_always_the_first_question_and_blocks_import(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def workspace_factory(output_root: Path, run_id: str):
        events.append("workspace")
        return create_workspace(output_root, run_id)

    def importer(config):
        events.append("import")
        return config

    session = ChecklistSession(
        run_id="run-001",
        workspace_factory=workspace_factory,
    )

    assert session.next_prompt() == "请选择本次分析的结果导出文件夹。"
    with pytest.raises(RuntimeError, match="尚未选择结果导出文件夹"):
        session.run_import(importer)
    assert events == []

    session.select_output_root(tmp_path / "results")
    session.submit_checklist(valid_payload(tmp_path))
    session.run_import(importer)
    assert events == ["workspace", "import"]


def test_checklist_evidence_is_versioned_and_saved_in_step_directory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    workspace = create_workspace(output_root, run_id="run-001")
    payload = valid_payload(tmp_path)
    payload["output_root"] = str(output_root)
    result = validate_checklist(payload)
    now = datetime(2026, 7, 27, 15, 5)

    first = write_checklist_evidence(
        result,
        workspace=workspace,
        run_id="run-001",
        confirmed_at=now,
    )
    second = write_checklist_evidence(
        result,
        workspace=workspace,
        run_id="run-001",
        confirmed_at=now,
    )

    expected_parent = output_root / "操作步骤" / "01_环境与Checklist"
    assert all(path.parent == expected_parent for path in first)
    assert {path.suffix for path in first} == {".json", ".md"}
    assert all("ver.001" in path.name for path in first)
    assert all("ver.002" in path.name for path in second)
    checklist_json = next(path for path in first if "导入Checklist" in path.name and path.suffix == ".json")
    saved = json.loads(checklist_json.read_text(encoding="utf-8"))
    assert saved["run_id"] == "run-001"
    assert saved["confirmed_at"] == "2026-07-27T15:05:00"


def test_checklist_schema_declares_all_required_top_level_fields() -> None:
    schema_path = Path(__file__).parents[1] / "assets" / "checklist_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert set(schema["required"]) == {
        "output_root",
        "input_format",
        "input_files",
        "expected_sfreq",
        "file_order_confirmed",
        "same_animal_continuous_recording",
        "channels",
        "bipolar_formulas",
        "markers",
    }
