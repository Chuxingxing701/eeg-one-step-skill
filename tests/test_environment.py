from pathlib import Path
import os
import subprocess
import sys
import webbrowser

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.environment import (  # noqa: E402
    EnvironmentNotReadyError,
    assert_environment_ready,
    probe_environment,
    render_setup_guidance,
)


def fail_if_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("环境检测不得启动外部进程、安装器或浏览器")


def test_probe_environment_has_no_external_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", fail_if_called)
    monkeypatch.setattr(subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(os, "system", fail_if_called)
    monkeypatch.setattr(webbrowser, "open", fail_if_called)

    report = probe_environment()

    assert report.python_executable == Path(sys.executable)
    assert report.python_version == sys.version.split()[0]
    assert report.packages["mne"] is not None
    assert report.packages["numpy"] is not None
    assert report.packages["scipy"] is not None
    assert report.packages["pandas"] is not None
    assert report.packages["matplotlib"] is not None
    assert report.packages["pytest"] is not None
    assert report.ready is True


def test_missing_package_blocks_the_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eeg_pipeline import environment

    real_version = environment.metadata.version

    def fake_version(name: str) -> str:
        if name == "mne":
            raise environment.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(environment.metadata, "version", fake_version)
    report = probe_environment()

    assert report.ready is False
    assert report.missing_packages == ("mne",)
    with pytest.raises(EnvironmentNotReadyError, match="mne"):
        assert_environment_ready(report)


def test_missing_environment_guidance_is_manual_and_requests_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eeg_pipeline import environment

    real_version = environment.metadata.version

    def fake_version(name: str) -> str:
        if name == "mne":
            raise environment.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(environment.metadata, "version", fake_version)
    guidance = render_setup_guidance(probe_environment())

    assert "MNE" in guidance
    assert "Anaconda" in guidance
    assert "JupyterLab" in guidance
    assert "Python解释器完整路径" in guidance
    assert "环境名称" in guidance
    assert "Codex不会替你注册账号" in guidance
    assert "停止" in guidance


def test_ready_environment_guidance_reports_current_interpreter() -> None:
    report = probe_environment()

    guidance = render_setup_guidance(report)

    assert str(Path(sys.executable)) in guidance
    assert "环境检查通过" in guidance


def test_missing_git_blocks_antila_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from eeg_pipeline import environment

    real_executable_path = environment._executable_path

    def fake_executable_path(command: str):
        if command == "git":
            return None
        return real_executable_path(command)

    monkeypatch.setattr(environment, "_executable_path", fake_executable_path)
    report = probe_environment()

    assert report.missing_tools == ("git",)
    assert report.ready is False
    with pytest.raises(EnvironmentNotReadyError, match="git"):
        assert_environment_ready(report)
