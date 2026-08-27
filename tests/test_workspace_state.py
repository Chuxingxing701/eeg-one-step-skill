from datetime import datetime
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.provenance import (  # noqa: E402
    sha256_file,
    verify_manifest,
    write_manifest,
)
from eeg_pipeline.state import PipelineState  # noqa: E402
from eeg_pipeline.workspace import (  # noqa: E402
    create_workspace,
    next_versioned_path,
    write_bytes_exclusive,
)


EXPECTED_RUNTIME_DIRECTORIES = {
    "数据冻结",
    "数据冻结/01_输入文件清单",
    "数据冻结/02_直接拼接后",
    "数据冻结/03_双极参考后",
    "数据冻结/04_去伪迹前",
    "数据冻结/05_仅伪迹标注后",
    "数据冻结/06_滤波后",
    "操作步骤",
    "操作步骤/01_环境与Checklist",
    "操作步骤/02_导入与拼接",
    "操作步骤/03_Marker",
    "操作步骤/04_双极参考",
    "操作步骤/05_伪迹候选与审核",
    "操作步骤/06_工频检查与滤波",
    "操作步骤/07_Bandpower",
    "操作步骤/08_Antila分期",
    "操作步骤/09_逐图分析",
    "操作步骤/10_运行记录",
    "结果输出",
    "结果输出/数据表",
    "结果输出/图表",
    "结果输出/睡眠分期",
    "结果输出/睡眠分期/Antila",
    "结果输出/报告",
}


def test_create_workspace_builds_exact_tree_below_selected_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "用户选择的导出目录"

    workspace = create_workspace(output_root, run_id="run-001")

    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_dir()
    }
    assert actual == EXPECTED_RUNTIME_DIRECTORIES
    assert workspace.output_root == output_root.resolve()
    assert workspace.run_id == "run-001"
    assert all(
        path.resolve().is_relative_to(output_root.resolve())
        for path in workspace.directories.values()
    )


def test_next_versioned_path_increments_without_overwriting(tmp_path: Path) -> None:
    existing = tmp_path / "Bandpower逐窗结果-20260727-1430-ver.001.csv"
    existing.write_text("original", encoding="utf-8")
    now = datetime(2026, 7, 27, 14, 30)

    result = next_versioned_path(
        tmp_path,
        content_name="Bandpower逐窗结果",
        suffix=".csv",
        now=now,
    )

    assert result.name == "Bandpower逐窗结果-20260727-1430-ver.002.csv"
    assert existing.read_text(encoding="utf-8") == "original"


def test_exclusive_write_rejects_existing_frozen_path(tmp_path: Path) -> None:
    frozen = tmp_path / "双极参考后数据-20260727-1430-ver.001_raw.fif"
    frozen.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        write_bytes_exclusive(frozen, b"replacement")

    assert frozen.read_bytes() == b"original"


def test_pipeline_state_is_atomic_and_resumes_at_first_incomplete_gate(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "pipeline-state.json"
    evidence = tmp_path / "checklist.json"
    evidence.write_text('{"confirmed": true}', encoding="utf-8")
    state = PipelineState.create(
        state_path,
        run_id="run-001",
        gate_order=("environment", "checklist", "import"),
    )

    assert state.next_incomplete_gate == "environment"
    state.complete_gate(
        "environment",
        evidence=[evidence],
        user_decision="environment-confirmed",
    )

    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists()
    reloaded = PipelineState.load(state_path)
    assert reloaded.gates["environment"].status == "complete"
    assert reloaded.gates["environment"].user_decision == "environment-confirmed"
    assert reloaded.gates["environment"].evidence[0].sha256 == sha256_file(evidence)
    assert reloaded.next_incomplete_gate == "checklist"


def test_completed_gate_cannot_be_completed_twice(tmp_path: Path) -> None:
    state = PipelineState.create(
        tmp_path / "pipeline-state.json",
        run_id="run-001",
        gate_order=("environment",),
    )
    evidence = tmp_path / "environment.json"
    evidence.write_text("{}", encoding="utf-8")
    state.complete_gate("environment", evidence=[evidence])

    with pytest.raises(RuntimeError, match="已经完成"):
        state.complete_gate("environment", evidence=[evidence])


def test_manifest_detects_the_exact_mutated_file(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_manifest([first, second], manifest)

    assert verify_manifest(manifest) == []
    first.write_text("changed", encoding="utf-8")

    assert verify_manifest(manifest) == [first.resolve()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "sha256"
    assert len(payload["files"]) == 2
