#!/usr/bin/env python3
"""
Video-based nystagmus amplitude/frequency analysis.

The script uses a deliberately conservative pipeline:
1. Register each frame to a head/forehead anchor from the reference frame.
2. Track a small iris template inside a stabilized eye region.
3. Reject blinks and tracking jumps.
4. Estimate oscillation amplitude and frequency from 2-second windows and draw error bars.

Outputs are written under analysis_outputs/nystagmus_research3 by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import cv2
import matplotlib
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Roi = Tuple[int, int, int, int]

EXPERIMENT_ORDER = [
    "Experiment_0.MOV",
    "Experiment_1.MOV",
    "Experiment_2L.MOV",
    "Experiment_2R.MOV",
    "Experiment_2U.MOV",
    "Experiment_2D.MOV",
    "Experiment_3.MOV",
    "Experiment_4N.MOV",
    "Experiment_4F.MOV",
    "Experiment_5M.MOV",
    "Experiment_5N.MOV",
    "Experiment_6.MOV",
    "Experiment_7.MOV",
    "Experiment_8.MOV",
    "Experiment_9.MOV",
]

BAR_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#9D755D",
    "#BAB0AC",
    "#A0CBE8",
    "#FFBE7D",
    "#8CD17D",
    "#D4A6C8",
    "#F1A2A1",
]


@dataclass(frozen=True)
class VideoConfig:
    label: str
    condition: str
    group: str
    anchor: Roi
    eyes: Dict[str, Roi]
    notes: str = ""


VIDEO_CONFIGS: Dict[str, VideoConfig] = {
    "Experiment_0": VideoConfig(
        label="No nystagmus",
        condition="Control / no nystagmus",
        group="control",
        anchor=(500, 300, 820, 380),
        eyes={"L": (540, 590, 320, 190), "R": (900, 590, 330, 190)},
    ),
    "Experiment_1": VideoConfig(
        label="Baseline",
        condition="Looking straight ahead",
        group="gaze",
        anchor=(500, 600, 850, 300),
        eyes={"L": (650, 900, 290, 170), "R": (1010, 910, 300, 170)},
    ),
    "Experiment_2L": VideoConfig(
        label="Left gaze",
        condition="Looking left",
        group="gaze",
        anchor=(400, 560, 1100, 250),
        eyes={"L": (530, 700, 430, 190), "R": (1030, 740, 400, 210)},
    ),
    "Experiment_2R": VideoConfig(
        label="Right gaze",
        condition="Looking right",
        group="gaze",
        anchor=(300, 560, 1150, 280),
        eyes={"L": (500, 790, 400, 190), "R": (910, 780, 430, 200)},
    ),
    "Experiment_2U": VideoConfig(
        label="Up gaze",
        condition="Looking up",
        group="gaze",
        anchor=(450, 500, 1050, 260),
        eyes={"L": (610, 680, 340, 180), "R": (1040, 680, 350, 180)},
    ),
    "Experiment_2D": VideoConfig(
        label="Down gaze",
        condition="Looking down",
        group="gaze",
        anchor=(350, 560, 1150, 300),
        eyes={"L": (520, 850, 380, 210), "R": (940, 850, 420, 220)},
        notes="Iris is partly eyelid-occluded; amplitude estimate is lower confidence.",
    ),
    "Experiment_3": VideoConfig(
        label="Reading",
        condition="Reading text",
        group="functional",
        anchor=(450, 500, 1100, 320),
        eyes={"L": (450, 840, 360, 190), "R": (970, 830, 360, 190)},
    ),
    "Experiment_4N": VideoConfig(
        label="Near target",
        condition="Near target / accommodation",
        group="accommodation",
        anchor=(450, 570, 1050, 300),
        eyes={"L": (500, 810, 330, 180), "R": (960, 820, 360, 190)},
    ),
    "Experiment_4F": VideoConfig(
        label="Far target",
        condition="Far target / accommodation",
        group="accommodation",
        anchor=(600, 500, 850, 300),
        eyes={"L": (760, 800, 330, 190), "R": (1130, 800, 330, 190)},
    ),
    "Experiment_5M": VideoConfig(
        label="Morning",
        condition="Morning / time of day",
        group="time_of_day",
        anchor=(280, 560, 1100, 310),
        eyes={"L": (370, 850, 360, 200), "R": (820, 850, 360, 210)},
    ),
    "Experiment_5N": VideoConfig(
        label="Evening",
        condition="Evening / time of day",
        group="time_of_day",
        anchor=(250, 560, 1100, 330),
        eyes={"L": (280, 870, 360, 200), "R": (770, 880, 390, 200)},
    ),
    "Experiment_6": VideoConfig(
        label="Screen use",
        condition="Before/after screen use",
        group="visual_strain",
        anchor=(360, 520, 1100, 280),
        eyes={"L": (450, 780, 330, 190), "R": (900, 780, 330, 190)},
    ),
    "Experiment_7": VideoConfig(
        label="Physiology",
        condition="Rest/exercise state",
        group="physiology",
        anchor=(350, 520, 1200, 280),
        eyes={"L": (500, 840, 400, 190), "R": (1030, 820, 400, 190)},
    ),
    "Experiment_8": VideoConfig(
        label="Glasses",
        condition="With glasses / optical correction",
        group="optical_correction",
        anchor=(400, 500, 1100, 250),
        eyes={"L": (460, 740, 350, 180), "R": (930, 740, 440, 190)},
        notes="Glasses glare and frame edges reduce tracking confidence.",
    ),
    "Experiment_9": VideoConfig(
        label="Optokinetic",
        condition="Moving stripes / optokinetic response",
        group="optokinetic",
        anchor=(350, 550, 1100, 300),
        eyes={"L": (420, 840, 360, 190), "R": (870, 840, 390, 190)},
    ),
}


def clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    x, y, w, h = [int(v) for v in roi]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def scale_roi(roi: Roi, scale: float) -> Roi:
    return tuple(int(round(v * scale)) for v in roi)  # type: ignore[return-value]


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def prep_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def robust_mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)) * 1.4826)


def dark_center(roi_img: np.ndarray) -> Tuple[float, float]:
    gray = prep_gray(roi_img).astype(np.float32)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    spatial = np.exp(-(((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.55)) ** 2))
    q = np.percentile(gray, 25)
    weights = np.clip(q - gray, 0, None) ** 2 * spatial
    if float(weights.sum()) <= 1e-6:
        return w / 2, h / 2
    return float((weights * xx).sum() / weights.sum()), float((weights * yy).sum() / weights.sum())


def interpolate_series(values: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    idx = np.arange(len(values))
    good = valid & np.isfinite(values)
    if int(good.sum()) < 8:
        return None
    filled = np.interp(idx, idx[good], values[good])
    return filled


def bandpass(values: np.ndarray, fps: float, low_hz: float = 0.4, high_hz: float = 8.0) -> np.ndarray:
    nyq = fps / 2.0
    high = min(high_hz, nyq * 0.85)
    low = min(low_hz, high * 0.5)
    if high <= low or len(values) < max(24, int(fps * 3)):
        return values - np.median(values)
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    try:
        return filtfilt(b, a, values)
    except ValueError:
        return values - np.median(values)


def robust_half_peak_to_peak(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float((np.percentile(values, 95) - np.percentile(values, 5)) / 2.0)


def dominant_frequency_hz(x_band: np.ndarray, y_band: np.ndarray, fps: float) -> Tuple[float, float]:
    """Estimate dominant oscillation frequency and a simple peak clarity ratio."""
    x_band = np.asarray(x_band, dtype=float)
    y_band = np.asarray(y_band, dtype=float)
    finite = np.isfinite(x_band) & np.isfinite(y_band)
    if int(finite.sum()) < max(12, int(round(fps * 1.25))):
        return float("nan"), float("nan")

    xy = np.column_stack([x_band[finite], y_band[finite]])
    xy -= np.nanmean(xy, axis=0)
    if float(np.nanstd(xy)) < 1e-6:
        return float("nan"), float("nan")

    cov = np.cov(xy, rowvar=False)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        signal = xy @ axis
    except np.linalg.LinAlgError:
        signal = xy[:, 0] if np.nanstd(xy[:, 0]) >= np.nanstd(xy[:, 1]) else xy[:, 1]

    signal -= np.nanmean(signal)
    if len(signal) < 8:
        return float("nan"), float("nan")
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    band = (freqs >= 0.5) & (freqs <= min(6.5, fps * 0.45))
    if int(band.sum()) < 2:
        return float("nan"), float("nan")

    band_indices = np.where(band)[0]
    peak_index = int(band_indices[np.argmax(power[band])])
    peak_freq = float(freqs[peak_index])

    if 0 < peak_index < len(power) - 1:
        left = power[peak_index - 1]
        center = power[peak_index]
        right = power[peak_index + 1]
        denom = left - 2 * center + right
        if abs(float(denom)) > 1e-12:
            offset = 0.5 * (left - right) / denom
            peak_freq = float(freqs[peak_index] + offset * (freqs[1] - freqs[0]))
            peak_freq = float(np.clip(peak_freq, freqs[band_indices[0]], freqs[band_indices[-1]]))

    background = float(np.median(power[band]))
    clarity = float(power[peak_index] / background) if background > 1e-12 else float("nan")
    return peak_freq, clarity


def periodic_amplitude_px(x_band: np.ndarray, y_band: np.ndarray, fps: float, frequency_hz: float) -> float:
    """Fit one sinusoid at the detected oscillation frequency along the dominant motion axis."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        return float("nan")
    x_band = np.asarray(x_band, dtype=float)
    y_band = np.asarray(y_band, dtype=float)
    finite = np.isfinite(x_band) & np.isfinite(y_band)
    if int(finite.sum()) < max(12, int(round(fps * 1.25))):
        return float("nan")

    xy = np.column_stack([x_band[finite], y_band[finite]])
    xy -= np.nanmean(xy, axis=0)
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(xy, rowvar=False))
        axis = eigvecs[:, int(np.argmax(eigvals))]
        signal = xy @ axis
    except np.linalg.LinAlgError:
        signal = xy[:, 0] if np.nanstd(xy[:, 0]) >= np.nanstd(xy[:, 1]) else xy[:, 1]

    signal -= np.nanmean(signal)
    t = np.arange(len(signal), dtype=float) / fps
    design = np.column_stack(
        [
            np.sin(2.0 * np.pi * frequency_hz * t),
            np.cos(2.0 * np.pi * frequency_hz * t),
            np.ones(len(signal)),
        ]
    )
    try:
        coef = np.linalg.lstsq(design, signal, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.hypot(coef[0], coef[1]))


def bootstrap_ci(values: Iterable[float], iterations: int = 4000) -> Tuple[float, float, float]:
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    center = float(np.median(vals))
    if len(vals) == 1:
        return center, center, center
    rng = np.random.default_rng(20260731)
    samples = rng.choice(vals, size=(iterations, len(vals)), replace=True)
    boot = np.median(samples, axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return center, float(lo), float(hi)


def reject_artifacts(values: np.ndarray, quality: np.ndarray, fps: float) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(quality)
    if int(valid.sum()) < 8:
        return valid

    q_med = np.median(quality[valid])
    q_mad = robust_mad(quality[valid])
    q_cut = max(0.18, min(0.55, q_med - 3.0 * q_mad))
    valid &= quality >= q_cut

    filled = interpolate_series(values, valid)
    if filled is None:
        return valid

    kernel = max(3, int(round(fps * 0.5)) | 1)
    smooth = medfilt(filled, kernel_size=kernel)
    residual = np.abs(values - smooth)
    res_mad = robust_mad(residual[valid])
    res_cut = max(4.0, 6.0 * res_mad)
    valid &= residual <= res_cut

    diffs = np.abs(np.diff(filled, prepend=filled[0]))
    diff_mad = robust_mad(diffs[valid])
    diff_cut = max(3.5, 8.0 * diff_mad)
    valid &= diffs <= diff_cut
    return valid


def analyze_video(
    video_path: Path,
    config: VideoConfig,
    output_dir: Path,
    analysis_scale: float,
    target_fps: Optional[float],
) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = 1 if not target_fps else max(1, int(round(native_fps / target_fps)))
    fps = native_fps / stride

    ref_index = frame_count // 2
    ref_native = read_frame(cap, ref_index)
    if analysis_scale != 1.0:
        ref = cv2.resize(ref_native, None, fx=analysis_scale, fy=analysis_scale, interpolation=cv2.INTER_AREA)
    else:
        ref = ref_native
    height, width = ref.shape[:2]

    anchor = clamp_roi(scale_roi(config.anchor, analysis_scale), width, height)
    ax, ay, aw, ah = anchor
    ref_anchor = prep_gray(ref[ay : ay + ah, ax : ax + aw])

    eye_templates = {}
    initial_centers = {}
    for eye_name, native_roi in config.eyes.items():
        roi = clamp_roi(scale_roi(native_roi, analysis_scale), width, height)
        x, y, w, h = roi
        cx, cy = dark_center(ref[y : y + h, x : x + w])
        template_size = int(round(58 * analysis_scale))
        template_size = max(18, template_size | 1)
        tx = int(max(0, min(w - template_size, round(cx - template_size / 2))))
        ty = int(max(0, min(h - template_size, round(cy - template_size / 2))))
        templ = prep_gray(ref[y + ty : y + ty + template_size, x + tx : x + tx + template_size])
        eye_templates[eye_name] = {
            "roi": roi,
            "template": templ,
            "template_size": template_size,
        }
        initial_centers[eye_name] = (tx + template_size / 2.0, ty + template_size / 2.0)

    search = int(round(170 * analysis_scale))
    rows: List[Dict[str, float]] = []
    frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame_native = cap.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue

        if analysis_scale != 1.0:
            frame = cv2.resize(frame_native, None, fx=analysis_scale, fy=analysis_scale, interpolation=cv2.INTER_AREA)
        else:
            frame = frame_native

        sx = max(0, ax - search)
        sy = max(0, ay - search)
        ex = min(width, ax + aw + search)
        ey = min(height, ay + ah + search)
        search_img = prep_gray(frame[sy:ey, sx:ex])
        head_score = float("nan")
        dx = dy = 0.0
        if search_img.shape[0] >= ref_anchor.shape[0] and search_img.shape[1] >= ref_anchor.shape[1]:
            match = cv2.matchTemplate(search_img, ref_anchor, cv2.TM_CCOEFF_NORMED)
            _, head_score, _, loc = cv2.minMaxLoc(match)
            dx = float(loc[0] - (ax - sx))
            dy = float(loc[1] - (ay - sy))

        row: Dict[str, float] = {
            "frame": float(frame_index),
            "time_s": float(frame_index / native_fps),
            "head_dx_px": float(dx / analysis_scale),
            "head_dy_px": float(dy / analysis_scale),
            "head_score": float(head_score),
        }

        for eye_name, eye_spec in eye_templates.items():
            x, y, w, h = eye_spec["roi"]
            template = eye_spec["template"]
            template_size = eye_spec["template_size"]
            xx = int(round(x + dx))
            yy = int(round(y + dy))
            xx, yy, ww, hh = clamp_roi((xx, yy, w, h), width, height)
            crop = prep_gray(frame[yy : yy + hh, xx : xx + ww])
            if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
                row[f"{eye_name}_x_px"] = float("nan")
                row[f"{eye_name}_y_px"] = float("nan")
                row[f"{eye_name}_quality"] = float("nan")
                continue
            match = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, eye_score, _, eye_loc = cv2.minMaxLoc(match)
            cx = eye_loc[0] + template_size / 2.0
            cy = eye_loc[1] + template_size / 2.0
            row[f"{eye_name}_x_px"] = float(cx / analysis_scale)
            row[f"{eye_name}_y_px"] = float(cy / analysis_scale)
            row[f"{eye_name}_quality"] = float(eye_score)
        rows.append(row)
        frame_index += 1

    cap.release()
    if not rows:
        raise RuntimeError(f"No frames analyzed for {video_path}")

    trace_path = output_dir / "traces" / f"{video_path.stem}_trace.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with trace_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    window_seconds = 2.0
    window_size = max(8, int(round(window_seconds * fps)))
    eye_window_rows: List[Dict[str, float]] = []
    full_trace = {}
    times = np.array([r["time_s"] for r in rows], dtype=float)
    head_quality = np.array([r["head_score"] for r in rows], dtype=float)

    for eye_name in config.eyes.keys():
        x = np.array([r[f"{eye_name}_x_px"] for r in rows], dtype=float)
        y = np.array([r[f"{eye_name}_y_px"] for r in rows], dtype=float)
        q = np.array([r[f"{eye_name}_quality"] for r in rows], dtype=float)
        joint_quality = np.minimum(q, head_quality)
        valid_x = reject_artifacts(x, joint_quality, fps)
        valid_y = reject_artifacts(y, joint_quality, fps)
        valid = valid_x & valid_y & np.isfinite(x) & np.isfinite(y)

        filled_x = interpolate_series(x, valid)
        filled_y = interpolate_series(y, valid)
        if filled_x is None or filled_y is None:
            full_trace[eye_name] = {
                "valid": valid,
                "x_band": np.full_like(x, np.nan),
                "y_band": np.full_like(y, np.nan),
            }
            continue

        x_band = bandpass(filled_x, fps)
        y_band = bandpass(filled_y, fps)
        full_trace[eye_name] = {"valid": valid, "x_band": x_band, "y_band": y_band}

        for start in range(0, len(rows) - window_size + 1, window_size):
            end = start + window_size
            valid_fraction = float(valid[start:end].sum() / window_size)
            if valid_fraction < 0.70:
                continue
            q_window = q[start:end][valid[start:end]]
            head_window = head_quality[start:end][valid[start:end]]
            median_eye_quality = float(np.nanmedian(q_window))
            median_head_quality = float(np.nanmedian(head_window))
            if median_eye_quality < 0.80 or median_head_quality < 0.85:
                continue

            xb = x_band[start:end][valid[start:end]]
            yb = y_band[start:end][valid[start:end]]
            xb_full = x_band[start:end]
            yb_full = y_band[start:end]
            ax_amp = robust_half_peak_to_peak(xb)
            ay_amp = robust_half_peak_to_peak(yb)
            radial_amp = math.sqrt(ax_amp * ax_amp + ay_amp * ay_amp)
            dominant_freq, frequency_peak_ratio = dominant_frequency_hz(xb_full, yb_full, fps)
            periodic_amp = periodic_amplitude_px(xb_full, yb_full, fps, dominant_freq)
            eye_window_rows.append(
                {
                    "file": video_path.name,
                    "condition": config.condition,
                    "eye": eye_name,
                    "window_start_s": float(times[start]),
                    "window_end_s": float(times[end - 1]),
                    "valid_fraction": valid_fraction,
                    "horizontal_amp_px": ax_amp,
                    "vertical_amp_px": ay_amp,
                    "radial_amp_px": radial_amp,
                    "periodic_amp_px": periodic_amp,
                    "dominant_frequency_hz": dominant_freq,
                    "frequency_peak_ratio": frequency_peak_ratio,
                    "mean_x_px": float(np.nanmean(x[start:end][valid[start:end]])),
                    "mean_y_px": float(np.nanmean(y[start:end][valid[start:end]])),
                    "median_eye_quality": median_eye_quality,
                    "median_head_quality": median_head_quality,
                }
            )

    draw_trace_plot(video_path.stem, config, output_dir, times, full_trace)
    draw_qc_frame(video_path, config, output_dir, ref_native, ref_index, initial_centers, analysis_scale)

    radial = [r["radial_amp_px"] for r in eye_window_rows]
    horizontal = [r["horizontal_amp_px"] for r in eye_window_rows]
    vertical = [r["vertical_amp_px"] for r in eye_window_rows]
    radial_median, radial_lo, radial_hi = bootstrap_ci(radial)
    horizontal_median, horizontal_lo, horizontal_hi = bootstrap_ci(horizontal)
    vertical_median, vertical_lo, vertical_hi = bootstrap_ci(vertical)
    frequency = [r["dominant_frequency_hz"] for r in eye_window_rows]
    frequency_median, frequency_lo, frequency_hi = bootstrap_ci(frequency)
    periodic = [r["periodic_amp_px"] for r in eye_window_rows]
    periodic_median, periodic_lo, periodic_hi = bootstrap_ci(periodic)

    return {
        "file": video_path.name,
        "label": config.label,
        "condition": config.condition,
        "group": config.group,
        "duration_s": frame_count / native_fps if native_fps else float("nan"),
        "native_fps": native_fps,
        "analyzed_fps": fps,
        "windows": len(eye_window_rows),
        "radial_amp_px": radial_median,
        "radial_ci_low_px": radial_lo,
        "radial_ci_high_px": radial_hi,
        "horizontal_amp_px": horizontal_median,
        "horizontal_ci_low_px": horizontal_lo,
        "horizontal_ci_high_px": horizontal_hi,
        "vertical_amp_px": vertical_median,
        "vertical_ci_low_px": vertical_lo,
        "vertical_ci_high_px": vertical_hi,
        "periodic_amp_px": periodic_median,
        "periodic_ci_low_px": periodic_lo,
        "periodic_ci_high_px": periodic_hi,
        "dominant_frequency_hz": frequency_median,
        "frequency_ci_low_hz": frequency_lo,
        "frequency_ci_high_hz": frequency_hi,
        "notes": config.notes,
        "window_rows": eye_window_rows,
    }


def draw_trace_plot(
    stem: str,
    config: VideoConfig,
    output_dir: Path,
    times: np.ndarray,
    full_trace: Dict[str, Dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for eye_name, trace in full_trace.items():
        valid = trace["valid"]
        x_band = trace["x_band"]
        y_band = trace["y_band"]
        x_plot = np.where(valid, x_band, np.nan)
        y_plot = np.where(valid, y_band, np.nan)
        axes[0].plot(times, x_plot, linewidth=0.9, label=f"{eye_name} eye")
        axes[1].plot(times, y_plot, linewidth=0.9, label=f"{eye_name} eye")
    axes[0].set_title(f"{stem}: stabilized oscillation trace")
    axes[0].set_ylabel("horizontal px")
    axes[1].set_ylabel("vertical px")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.axhline(0, color="#777777", linewidth=0.6)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = output_dir / "plots" / f"{stem}_trace.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_qc_frame(
    video_path: Path,
    config: VideoConfig,
    output_dir: Path,
    ref_native: np.ndarray,
    ref_index: int,
    initial_centers: Dict[str, Tuple[float, float]],
    analysis_scale: float,
) -> None:
    frame = ref_native.copy()
    ax, ay, aw, ah = config.anchor
    cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (255, 120, 0), 3)
    cv2.putText(frame, "head alignment", (ax, max(25, ay - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)

    for eye_name, roi in config.eyes.items():
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 0), 3)
        cx, cy = initial_centers[eye_name]
        center = (int(round(x + cx / analysis_scale)), int(round(y + cy / analysis_scale)))
        cv2.drawMarker(frame, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3)
        cv2.putText(frame, f"{eye_name} iris template", (x, max(25, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 0), 2)

    cv2.putText(
        frame,
        f"{video_path.name}, reference frame {ref_index}",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    path = output_dir / "qc_frames" / f"{video_path.stem}_qc.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def draw_summary_plot(summary_rows: List[Dict[str, object]], output_dir: Path) -> None:
    rows = [r for r in summary_rows if r["group"] == "gaze" and np.isfinite(float(r["periodic_amp_px"]))]
    order = ["Baseline", "Left gaze", "Right gaze", "Up gaze", "Down gaze"]
    rows.sort(key=lambda row: order.index(str(row["label"])) if str(row["label"]) in order else 999)

    labels = [str(r["label"]) for r in rows]
    y = np.array([float(r["periodic_amp_px"]) for r in rows])
    lo = np.array([float(r["periodic_ci_low_px"]) for r in rows])
    hi = np.array([float(r["periodic_ci_high_px"]) for r in rows])
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    ax.bar(labels, y, yerr=yerr, capsize=6, color=colors[: len(rows)], edgecolor="#222222", linewidth=0.8)
    ax.set_ylabel("periodic oscillation amplitude (pixels)")
    ax.set_title("Gaze dependence: stabilized periodic nystagmus amplitude")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for idx, row in enumerate(rows):
        note = str(row.get("notes") or "")
        if note:
            ax.text(idx, y[idx] + yerr[1, idx] + 0.8, "lower confidence", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = output_dir / "gaze_amplitude_errorbars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_all_experiments_plot(summary_rows: List[Dict[str, object]], output_dir: Path) -> None:
    rows = [r for r in summary_rows if np.isfinite(float(r["periodic_amp_px"]))]
    rows.sort(key=lambda row: EXPERIMENT_ORDER.index(str(row["file"])) if str(row["file"]) in EXPERIMENT_ORDER else 999)

    labels = [f"{str(r['file']).replace('Experiment_', '').replace('.MOV', '')}\n{r['label']}" for r in rows]
    y = np.array([float(r["periodic_amp_px"]) for r in rows])
    lo = np.array([float(r["periodic_ci_low_px"]) for r in rows])
    hi = np.array([float(r["periodic_ci_high_px"]) for r in rows])
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.bar(labels, y, yerr=yerr, capsize=5, color=BAR_COLORS[: len(rows)], edgecolor="#222222", linewidth=0.7)
    ax.set_ylabel("periodic oscillation amplitude (pixels)")
    ax.set_title("All experiments: stabilized periodic nystagmus amplitude")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    for idx, row in enumerate(rows):
        if row.get("notes"):
            ax.text(idx, y[idx] + yerr[1, idx] + 0.35, "*", ha="center", va="bottom", fontsize=14)
    fig.text(0.01, 0.01, "* lower-confidence tracking condition", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = output_dir / "all_experiments_amplitude_errorbars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_all_experiments_frequency_plot(summary_rows: List[Dict[str, object]], output_dir: Path) -> None:
    rows = [r for r in summary_rows if np.isfinite(float(r["dominant_frequency_hz"]))]
    rows.sort(key=lambda row: EXPERIMENT_ORDER.index(str(row["file"])) if str(row["file"]) in EXPERIMENT_ORDER else 999)

    labels = [f"{str(r['file']).replace('Experiment_', '').replace('.MOV', '')}\n{r['label']}" for r in rows]
    y = np.array([float(r["dominant_frequency_hz"]) for r in rows])
    lo = np.array([float(r["frequency_ci_low_hz"]) for r in rows])
    hi = np.array([float(r["frequency_ci_high_hz"]) for r in rows])
    yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.bar(labels, y, yerr=yerr, capsize=5, color=BAR_COLORS[: len(rows)], edgecolor="#222222", linewidth=0.7)
    ax.set_ylabel("oscillation frequency (Hz)")
    ax.set_title("All experiments: stabilized nystagmus frequency")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    for idx, row in enumerate(rows):
        if row.get("notes"):
            ax.text(idx, y[idx] + yerr[1, idx] + 0.08, "*", ha="center", va="bottom", fontsize=14)
    fig.text(0.01, 0.01, "* lower-confidence tracking condition", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = output_dir / "all_experiments_frequency_errorbars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csvs(summary_rows: List[Dict[str, object]], window_rows: List[Dict[str, float]], output_dir: Path) -> None:
    summary_path = output_dir / "nystagmus_amplitude_summary.csv"
    summary_fields = [
        "file",
        "label",
        "condition",
        "group",
        "duration_s",
        "native_fps",
        "analyzed_fps",
        "windows",
        "radial_amp_px",
        "radial_ci_low_px",
        "radial_ci_high_px",
        "horizontal_amp_px",
        "horizontal_ci_low_px",
        "horizontal_ci_high_px",
        "vertical_amp_px",
        "vertical_ci_low_px",
        "vertical_ci_high_px",
        "periodic_amp_px",
        "periodic_ci_low_px",
        "periodic_ci_high_px",
        "dominant_frequency_hz",
        "frequency_ci_low_hz",
        "frequency_ci_high_hz",
        "notes",
    ]
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in summary_fields})

    windows_path = output_dir / "nystagmus_window_measurements.csv"
    window_fields = [
        "file",
        "condition",
        "eye",
        "window_start_s",
        "window_end_s",
        "valid_fraction",
        "horizontal_amp_px",
        "vertical_amp_px",
        "radial_amp_px",
        "periodic_amp_px",
        "dominant_frequency_hz",
        "frequency_peak_ratio",
        "mean_x_px",
        "mean_y_px",
        "median_eye_quality",
        "median_head_quality",
    ]
    with windows_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=window_fields)
        writer.writeheader()
        for row in window_rows:
            writer.writerow({field: row.get(field, "") for field in window_fields})

    frequency_path = output_dir / "nystagmus_frequency_summary.csv"
    frequency_fields = [
        "file",
        "label",
        "condition",
        "group",
        "duration_s",
        "native_fps",
        "analyzed_fps",
        "windows",
        "dominant_frequency_hz",
        "frequency_ci_low_hz",
        "frequency_ci_high_hz",
        "notes",
    ]
    with frequency_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frequency_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in frequency_fields})


def write_report(summary_rows: List[Dict[str, object]], output_dir: Path, source_dir: Path) -> None:
    gaze_rows = [r for r in summary_rows if r["group"] == "gaze" and np.isfinite(float(r["periodic_amp_px"]))]
    min_row = min(gaze_rows, key=lambda r: float(r["periodic_amp_px"])) if gaze_rows else None
    max_row = max(gaze_rows, key=lambda r: float(r["periodic_amp_px"])) if gaze_rows else None

    lines = [
        "# Nystagmus Video Analysis",
        "",
        f"Source directory: `{source_dir}`",
        "Supplemental lookup: missing experiment files are also checked in the current workspace directory.",
        "",
        "Method: each clip was stabilized to a head/forehead template, iris position was tracked inside stabilized eye windows, blink/tracking artifacts were rejected, and frequency was estimated from the dominant spectral peak in 2-second windows. The amplitude chart uses the fitted periodic component at that frequency, which avoids counting non-periodic tracking drift as nystagmus. Error bars are bootstrap 95% confidence intervals across usable eye/windows.",
        "",
        "Important limitation: true physical gaze angle in degrees cannot be recovered from these videos alone. The results identify the lowest-amplitude gaze condition from the recorded directions, not a calibrated clinical null angle.",
        "",
    ]
    if min_row:
        lines.extend(
            [
                "## Gaze-Dependence Result",
                "",
                f"Lowest measured periodic oscillation amplitude: **{min_row['label']}** (`{min_row['file']}`), {float(min_row['periodic_amp_px']):.2f} px with 95% CI {float(min_row['periodic_ci_low_px']):.2f}-{float(min_row['periodic_ci_high_px']):.2f} px.",
            ]
        )
    if max_row:
        lines.append(
            f"Highest measured periodic oscillation amplitude in this subset: **{max_row['label']}** (`{max_row['file']}`), {float(max_row['periodic_amp_px']):.2f} px with 95% CI {float(max_row['periodic_ci_low_px']):.2f}-{float(max_row['periodic_ci_high_px']):.2f} px."
        )
    lines.extend(
        [
            "",
            "## Summary Table",
            "",
            "| File | Condition | Periodic amp px | 95% CI px | Raw movement amp px | Frequency Hz | Windows | Notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['file']} | {row['condition']} | {float(row['periodic_amp_px']):.2f} | "
            f"{float(row['periodic_ci_low_px']):.2f}-{float(row['periodic_ci_high_px']):.2f} | "
            f"{float(row['radial_amp_px']):.2f} | {float(row['dominant_frequency_hz']):.2f} | "
            f"{int(row['windows'])} | {row.get('notes') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `gaze_amplitude_errorbars.png`: gaze-condition bar chart with 95% CI error bars.",
            "- `all_experiments_amplitude_errorbars.png`: all analyzed clips with 95% CI error bars.",
            "- `all_experiments_frequency_errorbars.png`: all analyzed clips' dominant oscillation frequency with 95% CI error bars.",
            "- `nystagmus_amplitude_summary.csv`: one summary row per clip.",
            "- `nystagmus_frequency_summary.csv`: frequency-focused summary rows.",
            "- `nystagmus_window_measurements.csv`: per-eye, per-window measurements used for error bars.",
            "- `plots/*_trace.png`: stabilized horizontal/vertical traces after artifact rejection.",
            "- `qc_frames/*_qc.jpg`: checked eye/head regions used for tracking.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../Expanded Nystagmus Research 3", help="Directory containing source videos")
    parser.add_argument("--output", default="analysis_outputs/nystagmus_research3", help="Output directory")
    parser.add_argument("--scale", type=float, default=0.5, help="Frame scale used for tracking")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Approximate analysis fps; set 0 for native")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_fps = None if args.target_fps <= 0 else args.target_fps

    summary_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, float]] = []
    for stem, config in VIDEO_CONFIGS.items():
        video_path = source_dir / f"{stem}.MOV"
        if not video_path.exists():
            video_path = Path.cwd() / f"{stem}.MOV"
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue
        print(f"Analyzing {video_path.name}...")
        result = analyze_video(video_path, config, output_dir, args.scale, target_fps)
        window_rows.extend(result.pop("window_rows"))  # type: ignore[arg-type]
        summary_rows.append(result)

    write_csvs(summary_rows, window_rows, output_dir)
    draw_summary_plot(summary_rows, output_dir)
    draw_all_experiments_plot(summary_rows, output_dir)
    draw_all_experiments_frequency_plot(summary_rows, output_dir)
    write_report(summary_rows, output_dir, source_dir)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
