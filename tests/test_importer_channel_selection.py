from pathlib import Path

import mne
import numpy as np

from eeg_pipeline.importer import import_and_concatenate
from eeg_pipeline.models import BipolarPair, MarkerSpec, RunConfig


def test_import_keeps_only_user_confirmed_bipolar_source_channels(tmp_path: Path) -> None:
    source = tmp_path / "seven_channels_raw.fif"
    channel_names = (
        "Channel1",
        "Channel2",
        "Channel3",
        "Channel4",
        "X",
        "Y",
        "Z",
    )
    data = np.arange(7 * 200, dtype=float).reshape(7, 200) * 1e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(
            channel_names,
            sfreq=100.0,
            ch_types=("eeg", "eeg", "eeg", "eeg", "misc", "misc", "misc"),
        ),
        verbose="ERROR",
    )
    raw.save(source, overwrite=False, verbose="ERROR")
    destination = tmp_path / "selected_channels_raw.fif"
    config = RunConfig(
        output_root=tmp_path / "results",
        input_files=(source,),
        expected_sfreq=100.0,
        file_order_confirmed=True,
        same_animal_continuous_recording=True,
        bipolar_pairs=(
            BipolarPair("EEG_bipolar", "Channel4", "Channel3", "eeg"),
            BipolarPair("EMG_bipolar", "Channel2", "Channel1", "emg"),
        ),
        markers=(
            MarkerSpec(
                "N2O_start",
                "1.0",
                "concatenated_seconds",
                None,
                "开始吸入N2O",
            ),
        ),
    )

    result = import_and_concatenate(config, destination)
    reopened = mne.io.read_raw_fif(destination, preload=True, verbose="ERROR")
    untouched = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")

    assert reopened.ch_names == ["Channel1", "Channel2", "Channel3", "Channel4"]
    assert reopened.get_channel_types() == ["emg", "emg", "eeg", "eeg"]
    np.testing.assert_allclose(reopened.get_data(), data[:4], rtol=1e-6, atol=1e-12)
    assert result.selected_channel_names == (
        "Channel1",
        "Channel2",
        "Channel3",
        "Channel4",
    )
    assert result.selected_channel_types == ("emg", "emg", "eeg", "eeg")
    assert result.excluded_channel_names == ("X", "Y", "Z")
    assert untouched.ch_names == list(channel_names)
