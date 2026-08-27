"""Write a source inventory and journal-redraw guide for EEG outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .provenance import sha256_file


@dataclass(frozen=True)
class DataArtifact:
    name: str
    path: Path
    purpose: str
    unit_or_key_field: str


def write_data_inventory(
    artifacts: tuple[DataArtifact, ...], output_directory: Path
) -> Path:
    directory = output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for artifact in artifacts:
        path = artifact.path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "name": artifact.name,
                "path": str(path),
                "purpose": artifact.purpose,
                "unit_or_key_field": artifact.unit_or_key_field,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    destination = directory / "数据获取清单.csv"
    if destination.exists():
        raise FileExistsError(destination)
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def write_journal_redraw_guide(
    artifacts: tuple[DataArtifact, ...], output_directory: Path
) -> Path:
    directory = output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "数据获取与期刊重绘说明.md"
    if destination.exists():
        raise FileExistsError(destination)
    lines = [
        "# 数据获取与期刊重绘说明",
        "",
        "报告PNG只用于汇报。期刊图必须从清单中的CSV重新提取。",
        "",
        "## 数据位置",
        "",
    ]
    for artifact in artifacts:
        lines.append(
            f"- {artifact.name}：{artifact.path.expanduser().resolve()}；用途：{artifact.purpose}；单位或关键字段：{artifact.unit_or_key_field}。"
        )
    lines.extend(
        [
            "",
            "## 重绘规则",
            "",
            "- 时间轴优先使用start_seconds、stop_seconds和midpoint_seconds。",
            "- 绝对频段功率的原始单位为V²；绘制µV²时乘以10^12。",
            "- 按Marker生成实验阶段，再按sleep_stage筛选清醒、非快速眼动睡眠和快速眼动睡眠。",
            "- Artifact、Uncertain与Boundary_Unscored保持独立；无效Bandpower保持NaN。",
            "- 统计单位按动物或实验记录定义，不把相邻窗口当作独立动物。",
            "- 每张期刊图在方法或补充材料中注明来源CSV、字段、单位、筛选阶段和汇总统计量。",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination

