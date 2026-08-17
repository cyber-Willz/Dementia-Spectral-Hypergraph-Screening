#!/usr/bin/env python3
"""
eeg_feature_extraction.py
==========================

Turns preprocessed resting-state EEG (BIDS format, e.g. OpenNeuro ds004504:
"A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal
Dementia and Healthy subjects", Miltiadous et al. 2023,
https://openneuro.org/datasets/ds004504) into the per-subject,
per-epoch connectivity-graph JSON that `dementia_spec_screen` (the Rust
crate) expects -- see its `src/types.rs`.

This script is EEG-specific but dataset-format-agnostic: it works on any
BIDS `*_eeg.set/.edf/.fif/.vhdr` recording MNE can read. It does NOT
download data itself -- ds004504 is CC0 but hosted on OpenNeuro's S3-backed
storage, not a plain git remote, so fetch it yourself first, e.g.:

    pip install openneuro-py
    openneuro-py download --dataset ds004504 --target ./ds004504
    # or, if you have the AWS CLI:
    aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./ds004504

Usage
-----
    python eeg_feature_extraction.py \
        --bids-root ./ds004504 \
        --derivatives \
        --out-dir ./subject_graphs \
        --band alpha \
        --epoch-seconds 5.0 \
        --connectivity coherence

Each `<out-dir>/<subject_id>.json` can then be fed straight to:

    cargo run --release --bin screen -- ./subject_graphs ./screening_report.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import coherence, hilbert

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def find_subjects(bids_root: Path, use_derivatives: bool) -> list[str]:
    root = bids_root / "derivatives" if use_derivatives else bids_root
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist -- did you download the dataset first?")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def find_eeg_file(bids_root: Path, subject: str, use_derivatives: bool) -> Path:
    root = bids_root / "derivatives" if use_derivatives else bids_root
    eeg_dir = root / subject / "eeg"
    candidates = sorted(eeg_dir.glob(f"{subject}_task-*_eeg.*"))
    candidates = [c for c in candidates if c.suffix in (".set", ".edf", ".fif", ".vhdr", ".bdf")]
    if not candidates:
        raise FileNotFoundError(f"no EEG recording found under {eeg_dir}")
    return candidates[0]


def load_raw(path: Path):
    import mne

    mne.set_log_level("ERROR")
    if path.suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True)
    elif path.suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True)
    elif path.suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True)
    elif path.suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True)
    elif path.suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True)
    else:
        raise ValueError(f"unsupported EEG file type: {path.suffix}")
    raw.pick_types(eeg=True, exclude="bads")
    return raw


def band_filter(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    lo, hi = band
    nyq = sfreq / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def weighted_pli(analytic: np.ndarray) -> np.ndarray:
    """Weighted phase-lag index across channel pairs for one epoch's
    analytic (Hilbert-transformed) signal, shape [n_channels, n_samples].
    Returns an [n_channels, n_channels] symmetric matrix in [0, 1].
    Robust to volume-conduction zero-lag artifacts (unlike plain coherence),
    which is why it's the default -- see Vinck et al. 2011.
    """
    n_ch = analytic.shape[0]
    out = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            csd = analytic[i] * np.conj(analytic[j])
            im = np.imag(csd)
            num = np.abs(np.mean(np.abs(im) * np.sign(im)))
            den = np.mean(np.abs(im)) + 1e-12
            wpli = num / den
            out[i, j] = out[j, i] = wpli
    return out


def coherence_matrix(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Mean magnitude-squared coherence within `band`, across channel
    pairs, for one epoch. Shape [n_channels, n_channels], values in [0, 1].
    Simpler / more standard than wPLI, offered as a fallback."""
    n_ch = data.shape[0]
    out = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, cxy = coherence(data[i], data[j], fs=sfreq, nperseg=min(256, data.shape[-1]))
            mask = (f >= band[0]) & (f <= band[1])
            val = float(np.mean(cxy[mask])) if mask.any() else 0.0
            out[i, j] = out[j, i] = val
    return out


def extract_subject(
    raw,
    subject_id: str,
    band: str,
    epoch_seconds: float,
    connectivity: str,
    edge_threshold: float,
) -> dict:
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # [n_channels, n_samples]
    ch_names = raw.ch_names
    n_per_epoch = int(epoch_seconds * sfreq)
    n_epochs = data.shape[1] // n_per_epoch

    band_range = BANDS[band]
    filtered = band_filter(data, sfreq, band_range)

    epochs_out = []
    for e in range(n_epochs):
        seg = filtered[:, e * n_per_epoch : (e + 1) * n_per_epoch]
        if connectivity == "wpli":
            analytic = hilbert(seg, axis=-1)
            mat = weighted_pli(analytic)
        elif connectivity == "coherence":
            mat = coherence_matrix(seg, sfreq, band_range)
        else:
            raise ValueError(f"unknown connectivity metric: {connectivity}")

        edges = []
        n_ch = len(ch_names)
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                w = float(mat[i, j])
                if w >= edge_threshold:
                    edges.append({"source": ch_names[i], "target": ch_names[j], "weight": w})
        epochs_out.append({"epoch_index": e, "nodes": list(ch_names), "edges": edges})

    return {"subject_id": subject_id, "epochs": epochs_out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", type=Path, required=True)
    ap.add_argument("--derivatives", action="store_true", help="read from derivatives/ (preprocessed) instead of raw")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--band", choices=list(BANDS), default="alpha")
    ap.add_argument("--epoch-seconds", type=float, default=5.0)
    ap.add_argument("--connectivity", choices=["wpli", "coherence"], default="wpli")
    ap.add_argument("--edge-threshold", type=float, default=0.05, help="drop edges weaker than this")
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to these subject IDs (e.g. sub-001)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or find_subjects(args.bids_root, args.derivatives)

    for subject_id in subjects:
        eeg_path = find_eeg_file(args.bids_root, subject_id, args.derivatives)
        print(f"[{subject_id}] loading {eeg_path.name}")
        raw = load_raw(eeg_path)
        record = extract_subject(
            raw, subject_id, args.band, args.epoch_seconds, args.connectivity, args.edge_threshold
        )
        out_path = args.out_dir / f"{subject_id}.json"
        out_path.write_text(json.dumps(record))
        print(f"[{subject_id}] wrote {len(record['epochs'])} epochs -> {out_path}")


if __name__ == "__main__":
    main()
