"""Create the fixed runtime tree and versioned, non-overwriting paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import re


RUNTIME_DIRECTORIES = (
    "数据冻结/01_输入文件清单",
    "数据冻结/02_直接拼接后",
    "数据冻结/03_双极参考后",
    "数据冻结/04_去伪迹前",
    "数据冻结/05_仅伪迹标注后",
    "数据冻结/06_滤波后",
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
    "结果输出/数据表",
    "结果输出/图表",
    "结果输出/睡眠分期/Antila",
    "结果输出/报告",
)


@dataclass(frozen=True)
class WorkspacePaths:
    output_root: Path
    run_id: str
    directories: Mapping[str, Path]

    def directory(self, relative_path: str) -> Path:
        return self.directories[relative_path]


def _validate_root(output_root: Path) -> Path:
    if str(output_root).strip() in {"", "."}:
        raise ValueError("必须明确选择结果导出文件夹")
    resolved = output_root.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def create_workspace(output_root: Path, run_id: str) -> WorkspacePaths:
    """Create the exact output tree beneath a user-selected directory."""

    root = _validate_root(output_root)
    if not run_id.strip():
        raise ValueError("run_id不能为空")
    root.mkdir(parents=True, exist_ok=True)

    directories: dict[str, Path] = {}
    for relative in RUNTIME_DIRECTORIES:
        path = (root / Path(relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"输出目录越界：{relative}")
        path.mkdir(parents=True, exist_ok=True)
        directories[relative] = path

    return WorkspacePaths(
        output_root=root,
        run_id=run_id,
        directories=MappingProxyType(directories),
    )


def next_versioned_path(
    directory: Path,
    content_name: str,
    suffix: str,
    now: datetime,
) -> Path:
    """Return the next `ver.NNN` path for one content name and minute."""

    if not content_name.strip():
        raise ValueError("工作内容名称不能为空")
    normalized_suffix = (
        suffix
        if suffix.startswith((".", "_"))
        else f".{suffix}"
    )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d-%H%M")
    pattern = re.compile(
        rf"^{re.escape(content_name)}-{timestamp}-ver\.(\d{{3}})"
        rf"{re.escape(normalized_suffix)}$"
    )
    versions = []
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    return directory / (
        f"{content_name}-{timestamp}-ver.{version:03d}{normalized_suffix}"
    )


def write_bytes_exclusive(path: Path, data: bytes) -> Path:
    """Write a frozen artifact once and fail if the path already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
    return path
