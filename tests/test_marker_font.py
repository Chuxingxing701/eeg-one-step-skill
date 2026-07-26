from pathlib import Path
import sys
import warnings


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from eeg_pipeline.markers import ResolvedMarker, plot_marker_timeline  # noqa: E402


def test_marker_timeline_renders_chinese_without_missing_glyph_warning(
    tmp_path: Path,
) -> None:
    marker = ResolvedMarker(
        name="给药开始",
        onset_seconds=5.0,
        meaning="开始吸入笑气",
        original_value="5",
        original_basis="concatenated_seconds",
        source_file=None,
        source_offset_seconds=0.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        plot_marker_timeline([marker], 10.0, tmp_path / "实验Marker时间轴.png")
