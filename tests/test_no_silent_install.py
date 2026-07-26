from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).parents[1]
EXECUTABLE_SUFFIXES = {".py", ".ps1", ".bat", ".cmd"}


def test_executable_sources_contain_no_silent_install_commands() -> None:
    forbidden_patterns = (
        re.compile("pip" + r"\s+" + "install", re.IGNORECASE),
        re.compile("conda" + r"\s+" + "install", re.IGNORECASE),
        re.compile("mamba" + r"\s+" + "install", re.IGNORECASE),
        re.compile("winget" + r"\s+" + "install", re.IGNORECASE),
        re.compile("choco" + r"\s+" + "install", re.IGNORECASE),
        re.compile("Start" + "-Process" + r".*(?:\.exe|\.msi)", re.IGNORECASE),
    )
    violations: list[str] = []

    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXECUTABLE_SUFFIXES:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {pattern.pattern}")

    assert violations == []
