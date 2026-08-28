from __future__ import annotations

from pathlib import Path

from eeg_pipeline.data_guide import DataArtifact, write_data_inventory, write_journal_redraw_guide


def test_data_inventory_and_redraw_guide_name_every_source(tmp_path: Path) -> None:
    artifacts = (
        DataArtifact("原始EDF", tmp_path / "input.edf", "原始记录", "none"),
        DataArtifact("滤波FIF", tmp_path / "filtered_raw.fif", "Antila输入", "none"),
        DataArtifact("Bandpower逐窗", tmp_path / "bandpower.csv", "频段功率", "V2"),
        DataArtifact("Antila逐窗", tmp_path / "antila.csv", "睡眠分期", "sleep_stage"),
    )
    for item in artifacts:
        item.path.write_text("fixture", encoding="utf-8")
    inventory = write_data_inventory(artifacts, tmp_path / "output")
    guide = write_journal_redraw_guide(artifacts, tmp_path / "output")
    text = guide.read_text(encoding="utf-8")
    assert inventory.exists()
    for required in ("原始EDF", "滤波FIF", "Bandpower逐窗", "Antila逐窗", "时间轴", "单位", "期刊"):
        assert required in text

