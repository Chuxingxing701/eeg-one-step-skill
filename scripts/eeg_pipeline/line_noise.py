"""Inspect and filter 50/100 Hz line noise with explicit user gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import csv
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from .workspace import next_versioned_path, write_bytes_exclusive


@dataclass(frozen=True)
class LineNoiseConfig:
    channel_names: tuple[str, ...] = ("EEG_bipolar", "EMG_bipolar")
    psd_window_seconds: float = 3.0
    overlap_fraction: float = 0.5
    fmin: float = 0.5
    fmax: float = 120.0
    required_frequency_hz: float = 50.0
    optional_frequency_hz: float = 100.0
    notch_method: str = "fir"
    notch_phase: str = "zero"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    trans_bandwidth_hz: float = 1.0


@dataclass(frozen=True)
class StepNotice:
    step_number: int
    total_steps: int
    title: str
    method: str
    purpose: str
    output: str

    def render(self) -> str:
        return (
            f"[步骤 {self.step_number}/{self.total_steps}] {self.title}\n"
            f"方法：{self.method}\n"
            f"目的：{self.purpose}\n"
            f"产出：{self.output}"
        )


@dataclass(frozen=True)
class LineNoiseInputQC:
    passed: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PowerSpectrum:
    frequencies_hz: np.ndarray
    values: np.ndarray
    channel_names: tuple[str, ...]
    n_per_seg: int
    n_overlap: int
    n_fft: int
    window_name: str = "hann"
    average: str = "mean"
    reject_by_annotation: bool = True


@dataclass(frozen=True)
class LineNoiseEvidence:
    channel_name: str
    target_frequency_hz: float
    nearest_bin_hz: float
    peak_psd: float
    local_baseline_psd: float
    peak_to_local_ratio: float
    is_local_maximum: bool


@dataclass(frozen=True)
class PSDReviewBundle:
    psd_csv_path: Path
    evidence_csv_path: Path
    config_path: Path
    report_path: Path
    review_manifest_path: Path
    chart_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LineNoiseFilterQC:
    passed: bool
    errors: tuple[str, ...]
    n_times: int
    sfreq: float
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    annotations_equal: bool


@dataclass(frozen=True)
class LineNoiseFilterResult:
    raw: mne.io.BaseRaw
    frequencies_hz: tuple[float, ...]
    qc: LineNoiseFilterQC


def _validate_config(config: LineNoiseConfig) -> list[str]:
    errors: list[str] = []
    if not config.channel_names:
        errors.append("必须指定至少一个PSD通道")
    if len(config.channel_names) != len(set(config.channel_names)):
        errors.append("PSD通道不能重复")
    if config.psd_window_seconds <= 0:
        errors.append("PSD窗口必须大于0秒")
    if not 0 <= config.overlap_fraction < 1:
        errors.append("PSD重叠比例必须大于等于0且小于1")
    if config.fmin < 0 or config.fmax <= config.fmin:
        errors.append("PSD频率范围无效")
    if config.required_frequency_hz <= 0 or config.optional_frequency_hz <= 0:
        errors.append("工频目标必须为正数")
    if np.isclose(config.required_frequency_hz, config.optional_frequency_hz):
        errors.append("必选与可选工频目标不能相同")
    if config.notch_method != "fir":
        errors.append("本版本只允许已记录的FIR陷波方法")
    if config.notch_phase != "zero":
        errors.append("本版本只允许离线零相位陷波")
    return list(dict.fromkeys(errors))


def line_noise_step_notices(config: LineNoiseConfig) -> tuple[StepNotice, ...]:
    errors = _validate_config(config)
    if errors:
        raise ValueError("；".join(errors))
    return (
        StepNotice(
            1,
            7,
            "输入结构复核",
            "核对通道、类型、采样率、样本数及Annotations，不修改数据。",
            "确认当前文件可进入工频分析。",
            "输入QC结果。",
        ),
        StepNotice(
            2,
            7,
            "计算滤波前PSD",
            "采用论文报告的3秒Hann-Welch窗口与50%重叠；逐通道计算并排除BAD窗口。",
            "量化滤波前频谱及50/100 Hz局部峰。",
            "滤波前PSD与局部峰数值表。",
        ),
        StepNotice(
            3,
            7,
            "查看滤波前工频图",
            "输出完整PSD、45至55 Hz及95至105 Hz图，由Codex逐图核对数值。",
            "判断是否存在窄而尖的工频峰。",
            "三张滤波前图及逐图说明。",
        ),
        StepNotice(
            4,
            7,
            "确认100 Hz处理决定",
            "50 Hz按已批准要求滤除；100 Hz必须展示证据并由用户明确决定。",
            "防止程序擅自删除可能涉及高频生理信息的100 Hz成分。",
            "filter_100=True或False的用户决策记录。",
        ),
        StepNotice(
            5,
            7,
            "执行工频陷波",
            "采用MNE离线零相位FIR工程处理；论文未报告该陷波参数。",
            "滤除50 Hz及用户批准时的100 Hz窄带干扰。",
            "内存中的滤波后Raw及结构QC。",
        ),
        StepNotice(
            6,
            7,
            "计算滤波后PSD并对照",
            "继续使用同一论文PSD几何，输出相同三种图以保持可比性。",
            "检查目标峰是否降低，邻近频率是否异常受损。",
            "三张滤波后图、数值表和对照说明。",
        ),
        StepNotice(
            7,
            7,
            "冻结滤波结果",
            "使用不可覆盖FIF、Annotations、配置、软件版本和SHA256证据。",
            "形成可追溯的后续Bandpower输入。",
            "数据冻结/06_滤波后中的FIF及证据文件。",
        ),
    )


def line_noise_method_explanation(config: LineNoiseConfig) -> str:
    errors = _validate_config(config)
    if errors:
        raise ValueError("；".join(errors))
    return (
        "# 工频检查与滤波方法说明\n\n"
        "## 论文方法\n\n"
        f"论文报告的一般PSD采用{config.psd_window_seconds:g}秒Hann-Welch窗口、"
        f"{config.overlap_fraction * 100:g}%重叠。本Skill严格用这一几何计算"
        "滤波前后PSD，并对EEG和EMG分别显式选择通道。BAD区间只从PSD计算中"
        "排除，不删除样本。\n\n"
        "## 工程处理\n\n"
        "论文未报告陷波的滤波器类型、阶数、相位或宽度，因此不能把陷波称为"
        "论文参数。本Skill采用MNE的离线零相位FIR工程处理：50 Hz按用户批准"
        "要求滤除；100 Hz先展示95至105 Hz图和数值证据，再由用户明确决定。"
        "本阶段不执行0.5-100 Hz带通，因为该带通并非论文明确报告的方法。\n\n"
        "## 看图标准\n\n"
        "工频干扰通常表现为恰好位于50 Hz或100 Hz的窄而尖局部峰，明显高于"
        "两侧邻近频率，并可在多个通道重复出现。宽频抬升或仅一个宽峰不足以"
        "证明工频干扰。滤波后应看到目标窄峰降低，同时邻近频率总体形状保留；"
        "是否滤除100 Hz由用户判断，程序不自动下结论。\n"
    )


def _annotations_equal(left: mne.Annotations, right: mne.Annotations) -> bool:
    return (
        left.orig_time == right.orig_time
        and np.array_equal(left.description, right.description)
        and np.allclose(left.onset, right.onset, rtol=0.0, atol=1e-12)
        and np.allclose(left.duration, right.duration, rtol=0.0, atol=1e-12)
    )


def validate_line_noise_input(
    raw: mne.io.BaseRaw,
    config: LineNoiseConfig,
) -> LineNoiseInputQC:
    errors = _validate_config(config)
    missing = [name for name in config.channel_names if name not in raw.ch_names]
    if missing:
        errors.append(f"缺少PSD通道：{', '.join(missing)}")
    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        errors.append("采样率必须大于0")
    else:
        nyquist = sfreq / 2
        if config.fmax > nyquist:
            errors.append("PSD上限超过Nyquist频率")
        if config.required_frequency_hz >= nyquist:
            errors.append("50 Hz目标不低于Nyquist频率")
        if config.optional_frequency_hz >= nyquist:
            errors.append("100 Hz目标不低于Nyquist频率")
        if raw.n_times < round(config.psd_window_seconds * sfreq):
            errors.append("数据短于一个论文PSD窗口")
    return LineNoiseInputQC(
        passed=not errors,
        errors=tuple(dict.fromkeys(errors)),
    )


def compute_paper_psd(
    raw: mne.io.BaseRaw,
    config: LineNoiseConfig,
) -> PowerSpectrum:
    """Compute EEG and EMG separately with the paper-reported Welch geometry."""

    qc = validate_line_noise_input(raw, config)
    if not qc.passed:
        raise ValueError("；".join(qc.errors))
    sfreq = float(raw.info["sfreq"])
    n_per_seg = int(round(config.psd_window_seconds * sfreq))
    n_overlap = int(round(n_per_seg * config.overlap_fraction))
    rows = []
    shared_frequencies = None
    for channel_name in config.channel_names:
        single = raw.copy().pick([channel_name])
        spectrum = single.compute_psd(
            method="welch",
            fmin=config.fmin,
            fmax=config.fmax,
            n_fft=n_per_seg,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            window="hann",
            average="mean",
            picks="all",
            reject_by_annotation=True,
            verbose="ERROR",
        )
        values, frequencies = spectrum.get_data(
            picks="all",
            return_freqs=True,
        )
        row = np.asarray(values, dtype=float)
        if row.shape[0] != 1:
            raise RuntimeError(f"{channel_name} PSD通道数不是1")
        frequencies = np.asarray(frequencies, dtype=float)
        if shared_frequencies is None:
            shared_frequencies = frequencies
        elif not np.array_equal(shared_frequencies, frequencies):
            raise RuntimeError("EEG与EMG的PSD频率轴不一致")
        rows.append(row[0])
    if shared_frequencies is None:
        raise RuntimeError("未生成PSD频率轴")
    values = np.vstack(rows)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("PSD包含NaN或Inf，请检查有效非BAD数据长度")
    return PowerSpectrum(
        frequencies_hz=shared_frequencies,
        values=values,
        channel_names=config.channel_names,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        n_fft=n_per_seg,
    )


def measure_line_noise(
    psd: PowerSpectrum,
    targets_hz: tuple[float, ...],
) -> tuple[LineNoiseEvidence, ...]:
    evidence = []
    frequencies = psd.frequencies_hz
    for channel_index, channel_name in enumerate(psd.channel_names):
        values = psd.values[channel_index]
        for target in targets_hz:
            target_index = int(np.argmin(np.abs(frequencies - target)))
            nearest = float(frequencies[target_index])
            local_mask = (
                (frequencies >= target - 5.0)
                & (frequencies <= target + 5.0)
                & (np.abs(frequencies - target) > 1.0)
            )
            local_values = values[local_mask]
            if local_values.size == 0:
                raise ValueError(f"{target:g} Hz附近缺少可用基线频点")
            peak = float(values[target_index])
            baseline = float(np.median(local_values))
            ratio = peak / baseline if baseline > 0 else float("inf")
            neighbor_indices = [
                index
                for index in (target_index - 1, target_index + 1)
                if 0 <= index < values.size
            ]
            is_local_maximum = bool(
                neighbor_indices
                and all(peak > float(values[index]) for index in neighbor_indices)
            )
            evidence.append(
                LineNoiseEvidence(
                    channel_name=channel_name,
                    target_frequency_hz=float(target),
                    nearest_bin_hz=nearest,
                    peak_psd=peak,
                    local_baseline_psd=baseline,
                    peak_to_local_ratio=float(ratio),
                    is_local_maximum=is_local_maximum,
                )
            )
    return tuple(evidence)


@lru_cache(maxsize=1)
def _chinese_font() -> font_manager.FontProperties:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        if name in available:
            return font_manager.FontProperties(family=name)
    raise RuntimeError(
        "没有检测到可用中文字体。请用户手动安装中文字体后重试；"
        "Skill不会静默下载或安装字体。"
    )


def _plot_psd(
    psd: PowerSpectrum,
    frequency_range: tuple[float, float],
    title: str,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask = (
        (psd.frequencies_hz >= frequency_range[0])
        & (psd.frequencies_hz <= frequency_range[1])
    )
    if not np.any(mask):
        raise ValueError(f"图表频率范围没有PSD频点：{frequency_range}")
    figure, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    chinese_font = _chinese_font()
    colors = ("#1F4E79", "#B33A3A", "#2A7F62", "#6B5B95")
    for index, channel_name in enumerate(psd.channel_names):
        db_values = 10 * np.log10(
            np.maximum(psd.values[index, mask], np.finfo(float).tiny)
        )
        axis.plot(
            psd.frequencies_hz[mask],
            db_values,
            label=channel_name,
            color=colors[index % len(colors)],
            linewidth=1.1,
        )
    axis.set_xlabel("频率（Hz）", fontproperties=chinese_font)
    axis.set_ylabel("PSD（dB，V²/Hz）", fontproperties=chinese_font)
    axis.set_title(title, fontproperties=chinese_font)
    axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis.legend()
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def write_psd_review_bundle(
    psd: PowerSpectrum,
    directory: Path,
    *,
    stage_label: str,
    now: datetime,
) -> PSDReviewBundle:
    """Write one comparable three-chart PSD bundle without deciding 100 Hz."""

    if stage_label not in {"滤波前", "滤波后"}:
        raise ValueError("PSD阶段只能是滤波前或滤波后")
    directory = directory.resolve()
    psd_csv_path = next_versioned_path(
        directory,
        f"{stage_label}PSD数据",
        ".csv",
        now,
    )
    evidence_csv_path = next_versioned_path(
        directory,
        f"{stage_label}工频局部峰证据",
        ".csv",
        now,
    )
    config_path = next_versioned_path(
        directory,
        f"{stage_label}PSD配置",
        ".json",
        now,
    )
    report_path = next_versioned_path(
        directory,
        f"{stage_label}工频检查报告",
        ".md",
        now,
    )
    review_manifest_path = next_versioned_path(
        directory,
        f"{stage_label}PSD逐图审核清单",
        ".json",
        now,
    )
    chart_specs = (
        (
            "完整PSD",
            (float(psd.frequencies_hz[0]), float(psd.frequencies_hz[-1])),
            "显示EEG与EMG在整个分析频率范围内的功率谱分布。",
            "查看整体谱形、50/100 Hz位置及邻近频率是否存在异常抬升。",
        ),
        (
            "45至55HzPSD",
            (45.0, 55.0),
            "放大50 Hz附近的局部功率谱。",
            "查看50 Hz是否形成窄而尖的局部峰，并明显高于两侧频率。",
        ),
        (
            "95至105HzPSD",
            (95.0, 105.0),
            "放大100 Hz附近的局部功率谱。",
            "查看100 Hz是否形成窄而尖的局部峰；是否滤除必须由用户决定。",
        ),
    )
    chart_paths = []
    chart_manifest = []
    for content_name, frequency_range, meaning, look_for in chart_specs:
        path = next_versioned_path(
            directory,
            f"{stage_label}{content_name}",
            ".png",
            now,
        )
        _plot_psd(
            psd,
            frequency_range,
            f"{stage_label}{content_name}",
            path,
        )
        chart_paths.append(path)
        chart_manifest.append(
            {
                "chart_path": str(path),
                "source_data_paths": [str(psd_csv_path), str(evidence_csv_path)],
                "meaning": meaning,
                "look_for": look_for,
                "codex_review": {
                    "status": "pending",
                    "observations": "",
                    "numerical_evidence": "",
                    "chart_conclusion": "",
                    "limitations": "",
                },
            }
        )

    psd_fields = ["frequency_hz", *psd.channel_names]
    psd_rows = [
        {
            "frequency_hz": float(frequency),
            **{
                channel_name: float(psd.values[channel_index, frequency_index])
                for channel_index, channel_name in enumerate(psd.channel_names)
            },
        }
        for frequency_index, frequency in enumerate(psd.frequencies_hz)
    ]
    write_bytes_exclusive(psd_csv_path, _csv_bytes(psd_fields, psd_rows))
    evidence = measure_line_noise(psd, (50.0, 100.0))
    evidence_rows = [asdict(item) for item in evidence]
    write_bytes_exclusive(
        evidence_csv_path,
        _csv_bytes(list(evidence_rows[0]), evidence_rows),
    )
    config_payload = {
        "method_source": "论文报告的一般PSD方法",
        "window": psd.window_name,
        "n_per_seg": psd.n_per_seg,
        "n_overlap": psd.n_overlap,
        "n_fft": psd.n_fft,
        "average": psd.average,
        "reject_by_annotation": psd.reject_by_annotation,
        "frequency_resolution_hz": float(np.diff(psd.frequencies_hz).mean()),
        "channel_names": list(psd.channel_names),
        "automatic_100hz_verdict": None,
    }
    write_bytes_exclusive(
        config_path,
        (json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    report = (
        f"# {stage_label}工频检查报告\n\n"
        "- PSD方法：论文报告的3秒Hann-Welch窗口，50%重叠。\n"
        "- 完整PSD图表含义：查看整个分析范围内的总体谱形。\n"
        "- 45至55 Hz图表含义：检查50 Hz是否为窄而尖的局部峰。\n"
        "- 95至105 Hz图表含义：检查100 Hz是否为窄而尖的局部峰。\n"
        "- 100 Hz是否滤除必须由用户决定，程序不设置自动结论。\n"
        "- 每张图必须由Codex读取图片并与PSD/局部峰CSV交叉核对。\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    manifest = {
        "stage_label": stage_label,
        "charts": chart_manifest,
    }
    write_bytes_exclusive(
        review_manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return PSDReviewBundle(
        psd_csv_path=psd_csv_path,
        evidence_csv_path=evidence_csv_path,
        config_path=config_path,
        report_path=report_path,
        review_manifest_path=review_manifest_path,
        chart_paths=tuple(chart_paths),
    )


def apply_line_noise_filter(
    raw: mne.io.BaseRaw,
    config: LineNoiseConfig,
    *,
    filter_100: bool | None,
) -> LineNoiseFilterResult:
    """Apply required 50 Hz and explicitly decided 100 Hz engineering notches."""

    if filter_100 is None:
        raise ValueError("100 Hz尚未由用户决定，不能执行陷波")
    input_qc = validate_line_noise_input(raw, config)
    if not input_qc.passed:
        raise ValueError("；".join(input_qc.errors))
    frequencies = [config.required_frequency_hz]
    if filter_100:
        frequencies.append(config.optional_frequency_hz)
    source_annotations = raw.annotations.copy()
    source_names = tuple(raw.ch_names)
    source_types = tuple(raw.get_channel_types())
    source_n_times = int(raw.n_times)
    source_sfreq = float(raw.info["sfreq"])
    result = raw.copy().load_data()
    result.notch_filter(
        freqs=np.asarray(frequencies, dtype=float),
        picks=list(config.channel_names),
        filter_length="auto",
        notch_widths=None,
        trans_bandwidth=config.trans_bandwidth_hz,
        method=config.notch_method,
        phase=config.notch_phase,
        fir_window=config.fir_window,
        fir_design=config.fir_design,
        pad="reflect_limited",
        n_jobs=1,
        verbose="ERROR",
    )
    errors = []
    if tuple(result.ch_names) != source_names:
        errors.append("陷波后通道名称或顺序发生变化")
    if tuple(result.get_channel_types()) != source_types:
        errors.append("陷波后通道类型发生变化")
    if result.n_times != source_n_times:
        errors.append("陷波后样本数发生变化")
    if not np.isclose(result.info["sfreq"], source_sfreq):
        errors.append("陷波后采样率发生变化")
    annotations_equal = _annotations_equal(source_annotations, result.annotations)
    if not annotations_equal:
        errors.append("陷波后Annotations发生变化")
    qc = LineNoiseFilterQC(
        passed=not errors,
        errors=tuple(errors),
        n_times=int(result.n_times),
        sfreq=float(result.info["sfreq"]),
        channel_names=tuple(result.ch_names),
        channel_types=tuple(result.get_channel_types()),
        annotations_equal=annotations_equal,
    )
    if not qc.passed:
        raise RuntimeError("；".join(qc.errors))
    return LineNoiseFilterResult(
        raw=result,
        frequencies_hz=tuple(float(value) for value in frequencies),
        qc=qc,
    )


def write_filter_evidence(
    result: LineNoiseFilterResult,
    config: LineNoiseConfig,
    before_bundle: PSDReviewBundle,
    after_bundle: PSDReviewBundle,
    directory: Path,
    *,
    now: datetime,
) -> tuple[Path, Path, Path]:
    """Persist the engineering filter parameters, annotations, and structural QC."""

    directory = directory.resolve()
    config_path = next_versioned_path(
        directory,
        "工频陷波参数与用户决定",
        ".json",
        now,
    )
    annotations_path = next_versioned_path(
        directory,
        "滤波后Annotations",
        ".csv",
        now,
    )
    report_path = next_versioned_path(
        directory,
        "工频陷波执行报告",
        ".md",
        now,
    )
    expected_paths = (
        before_bundle.psd_csv_path,
        before_bundle.evidence_csv_path,
        *before_bundle.chart_paths,
        after_bundle.psd_csv_path,
        after_bundle.evidence_csv_path,
        *after_bundle.chart_paths,
    )
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    applied = list(result.frequencies_hz)
    filter_100 = config.optional_frequency_hz in result.frequencies_hz
    payload = {
        "filter_source": "工程处理，论文未报告陷波参数",
        "paper_psd_method": {
            "window_seconds": config.psd_window_seconds,
            "window": "hann",
            "overlap_fraction": config.overlap_fraction,
        },
        "applied_frequencies_hz": applied,
        "filter_100_user_decision": filter_100,
        "method": config.notch_method,
        "phase": config.notch_phase,
        "fir_window": config.fir_window,
        "fir_design": config.fir_design,
        "filter_length": "auto",
        "trans_bandwidth_hz": config.trans_bandwidth_hz,
        "notch_widths_hz": [frequency / 200.0 for frequency in applied],
        "notch_widths_source": "MNE默认规则：目标频率除以200",
        "pad": "reflect_limited",
        "channel_names": list(config.channel_names),
        "qc": asdict(result.qc),
        "before_bundle": {
            "psd_csv": str(before_bundle.psd_csv_path),
            "evidence_csv": str(before_bundle.evidence_csv_path),
            "charts": [str(path) for path in before_bundle.chart_paths],
        },
        "after_bundle": {
            "psd_csv": str(after_bundle.psd_csv_path),
            "evidence_csv": str(after_bundle.evidence_csv_path),
            "charts": [str(path) for path in after_bundle.chart_paths],
        },
    }
    write_bytes_exclusive(
        config_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    annotation_rows = [
        {
            "onset": float(onset),
            "duration": float(duration),
            "description": str(description),
        }
        for onset, duration, description in zip(
            result.raw.annotations.onset,
            result.raw.annotations.duration,
            result.raw.annotations.description,
        )
    ]
    write_bytes_exclusive(
        annotations_path,
        _csv_bytes(["onset", "duration", "description"], annotation_rows),
    )
    decision_text = "滤除" if filter_100 else "保留"
    report = (
        "# 工频陷波执行报告\n\n"
        "- 方法来源：工程处理，论文未报告陷波参数。\n"
        f"- 实际陷波频率：{', '.join(f'{value:g} Hz' for value in applied)}。\n"
        f"- 100 Hz用户决定：{decision_text}。\n"
        f"- 滤波实现：MNE {config.notch_method.upper()}，"
        f"{config.notch_phase}相位，{config.fir_window}窗，"
        f"{config.fir_design}设计。\n"
        "- 陷波宽度：MNE默认目标频率除以200；过渡带宽1 Hz。\n"
        f"- 样本数、通道和Annotations结构QC："
        f"{'通过' if result.qc.passed else '失败'}。\n"
        "- 滤波前后PSD均使用论文报告的3秒Hann-Welch窗口和50%重叠。\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    return config_path, annotations_path, report_path
