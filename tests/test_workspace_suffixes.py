from datetime import datetime
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.workspace import next_versioned_path  # noqa: E402


def test_versioned_path_preserves_raw_fif_composite_suffix(tmp_path: Path) -> None:
    result = next_versioned_path(
        tmp_path,
        content_name="双极参考后数据",
        suffix="_raw.fif",
        now=datetime(2026, 7, 27, 16, 20),
    )

    assert result.name == "双极参考后数据-20260727-1620-ver.001_raw.fif"
