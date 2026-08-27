"""Detect the user's EEG Python environment without changing it."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import shutil
import sys


REQUIRED_PACKAGES = (
    "mne",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pytest",
)


class EnvironmentNotReadyError(RuntimeError):
    """Raised when required user-managed packages are unavailable."""


@dataclass(frozen=True)
class EnvironmentReport:
    python_executable: Path
    python_version: str
    packages: Mapping[str, str | None]
    jupyter_lab_executable: Path | None
    git_executable: Path | None

    @property
    def missing_packages(self) -> tuple[str, ...]:
        return tuple(
            name for name, version in self.packages.items() if version is None
        )

    @property
    def missing_tools(self) -> tuple[str, ...]:
        return ("git",) if self.git_executable is None else ()

    @property
    def ready(self) -> bool:
        return not self.missing_packages and not self.missing_tools


def _executable_path(command: str) -> Path | None:
    resolved = shutil.which(command)
    return Path(resolved) if resolved else None


def probe_environment() -> EnvironmentReport:
    """Return interpreter and package facts using only in-process inspection."""

    packages: dict[str, str | None] = {}
    for package in REQUIRED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    return EnvironmentReport(
        python_executable=Path(sys.executable),
        python_version=sys.version.split()[0],
        packages=MappingProxyType(packages),
        jupyter_lab_executable=_executable_path("jupyter-lab"),
        git_executable=_executable_path("git"),
    )


def assert_environment_ready(report: EnvironmentReport) -> None:
    """Stop the workflow when required user-managed packages are missing."""

    if report.ready:
        return
    missing = ", ".join((*report.missing_packages, *report.missing_tools))
    raise EnvironmentNotReadyError(
        f"环境缺少必需包：{missing}。请按手动安装指南处理后再继续。"
    )


def render_setup_guidance(report: EnvironmentReport) -> str:
    """Render facts or a manual setup guide; never execute its commands."""

    if report.ready:
        package_lines = "\n".join(
            f"- {name}: {version}" for name, version in report.packages.items()
        )
        return (
            "环境检查通过。\n"
            f"- Python解释器：{report.python_executable}\n"
            f"- Python版本：{report.python_version}\n"
            f"{package_lines}\n"
            f"- JupyterLab：{report.jupyter_lab_executable or '未检测到命令行入口'}\n"
            f"- Git：{report.git_executable or '未检测到'}"
        )

    guide_path = Path(__file__).parents[2] / "references" / "environment-setup.md"
    guide = guide_path.read_text(encoding="utf-8")
    missing = "、".join((*report.missing_packages, *report.missing_tools))
    return (
        f"环境检查未通过，缺少：{missing}。正式流程现在停止。\n\n"
        f"当前Python解释器：{report.python_executable}\n\n"
        f"{guide}"
    )
