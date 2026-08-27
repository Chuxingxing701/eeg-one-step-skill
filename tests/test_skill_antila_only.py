from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", "evals"}


def test_skill_contains_no_retired_staging_source_or_documentation() -> None:
    forbidden = ("hil" + "bert", "ya" + "sa")
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        lowered_name = str(relative).lower()
        if any(token in lowered_name for token in forbidden):
            offenders.append(str(relative))
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        if any(token in text for token in forbidden):
            offenders.append(str(relative))
    assert offenders == []


def test_skill_contract_is_antila_only_and_has_data_redraw_outputs() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    for required in (
        "Antila Staging Gate",
        "tortugar/Lab",
        "bcb8dae1594e64a511545e34f6050e2a417c1f45",
        "Valid",
        "Artifact",
        "Boundary_Unscored",
        "数据获取",
        "期刊重绘",
        "本设计不可计算",
        "Antila分期结果",
        "Do not name, compare, recommend, or fall back to retired staging methods",
    ):
        assert required in skill
