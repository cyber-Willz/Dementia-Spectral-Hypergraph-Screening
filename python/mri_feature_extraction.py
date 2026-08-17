#!/usr/bin/env python3
"""
mri_feature_extraction.py
==========================

Turns a subject's FreeSurfer `recon-all` cortical parcellation output into
a single-subject **morphometric similarity network** (MSN, Seidlitz et al.
2018, "Morphometric Similarity Networks Detect Microscale Cortical
Organization and Predict Inter-Individual Cognitive Variation",
Neuron) -- one of the few structural-MRI connectivity measures that is
well-defined *within a single subject* from routine T1 output alone,
unlike population-level structural covariance networks.

This is the MRI-side counterpart to `eeg_feature_extraction.py`: it
outputs the same `dementia_spec_screen` JSON schema (`types::SubjectRecord`),
so the Rust pipeline treats an MRI-derived MSN and an EEG connectivity
graph identically.

Why this script does not download data itself
-----------------------------------------------
There is currently no EEG+MRI-paired dementia-subtype dataset openly
downloadable via a plain HTTP/git mirror the way ds004504 (EEG-only) is.
The MRI side is designed to work on **any** BIDS dataset with FreeSurfer
`derivatives/freesurfer/sub-*/stats/{lh,rh}.aparc.stats` output, including
gated cohorts you may already have access to under a data use agreement
(e.g. OASIS-3, ADNI) once you export/run `recon-all` locally -- this
script never talks to the network.

Input layout expected (standard `recon-all` layout, one dir per subject):

    <freesurfer-subjects-dir>/<subject_id>/stats/lh.aparc.stats
    <freesurfer-subjects-dir>/<subject_id>/stats/rh.aparc.stats

For a longitudinal cohort (multiple visits/sessions per subject, which is
what makes the downstream HMM's *dynamics* features meaningful rather than
a single static snapshot), point `--subjects-dir` at a directory laid out
as `<subject_id>_<session_id>/stats/...` (FreeSurfer's own longitudinal
naming convention) and pass `--sessions-glob` to group them; each session
becomes one "epoch" in `SubjectRecord`, in session order.

Usage
-----
    python mri_feature_extraction.py \
        --subjects-dir ./freesurfer_output \
        --out-dir ./subject_graphs_mri \
        --features ThickAvg SurfArea GrayVol MeanCurv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_FEATURES = ["ThickAvg", "SurfArea", "GrayVol", "MeanCurv", "GausCurv"]

STATS_HEADER_RE = re.compile(r"^#\s*ColHeaders\s+(.*)$")


def parse_aparc_stats(path: Path) -> dict[str, dict[str, float]]:
    """Parse one `{lh,rh}.aparc.stats` file into
    {roi_name: {column_name: value}}."""
    columns: list[str] | None = None
    rows: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            m = STATS_HEADER_RE.match(line)
            if m:
                columns = m.group(1).split()
            continue
        if not line.strip():
            continue
        parts = line.split()
        if columns is None or len(parts) != len(columns):
            continue
        row = dict(zip(columns, parts))
        roi = row["StructName"]
        rows[roi] = {k: float(v) for k, v in row.items() if k != "StructName"}
    return rows


def load_hemispheres(stats_dir: Path, hemi_prefix: bool = True) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for hemi, fname in (("lh", "lh.aparc.stats"), ("rh", "rh.aparc.stats")):
        path = stats_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        rois = parse_aparc_stats(path)
        for roi, vals in rois.items():
            key = f"{hemi}-{roi}" if hemi_prefix else roi
            out[key] = vals
    return out


def build_msn(rois: dict[str, dict[str, float]], features: list[str], edge_threshold: float) -> dict:
    """Build the morphometric similarity network: nodes = ROIs, edge weight
    = Pearson correlation (rescaled to [0, 1]) between two ROIs'
    z-scored multi-feature vectors, across the fixed `features` list.
    """
    names = sorted(rois.keys())
    matrix = np.array([[rois[n].get(f, np.nan) for f in features] for n in names], dtype=float)
    # z-score each feature column across ROIs (within-subject normalization,
    # standard for MSN construction so no single feature's scale dominates).
    mu = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0) + 1e-9
    z = (matrix - mu) / sd
    z = np.nan_to_num(z)

    n = len(names)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(z[i]) < 1e-9 or np.std(z[j]) < 1e-9:
                continue
            corr = float(np.corrcoef(z[i], z[j])[0, 1])
            w = (corr + 1.0) / 2.0  # rescale [-1, 1] -> [0, 1] for the Laplacian
            if w >= edge_threshold:
                edges.append({"source": names[i], "target": names[j], "weight": w})
    return {"nodes": names, "edges": edges}


def find_sessions(subjects_dir: Path, subject_id: str, sessions_glob: str | None) -> list[tuple[str, Path]]:
    """Returns [(session_label, stats_dir), ...] in sorted order. Falls
    back to a single "cross-sectional" session if no glob is given."""
    if sessions_glob is None:
        stats_dir = subjects_dir / subject_id / "stats"
        return [("ses-01", stats_dir)]
    sessions = []
    for d in sorted(subjects_dir.glob(sessions_glob.format(subject=subject_id))):
        stats_dir = d / "stats"
        if stats_dir.exists():
            sessions.append((d.name, stats_dir))
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    ap.add_argument("--edge-threshold", type=float, default=0.5)
    ap.add_argument(
        "--sessions-glob",
        default=None,
        help='e.g. "{subject}_ses-*" to group longitudinal visits; omit for single cross-sectional scans',
    )
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to these subject IDs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subject_ids = args.subjects
    else:
        seen = set()
        for d in sorted(args.subjects_dir.iterdir()):
            if not d.is_dir():
                continue
            sid = d.name.split("_ses-")[0].split("_")[0]
            seen.add(sid)
        subject_ids = sorted(seen)

    for subject_id in subject_ids:
        sessions = find_sessions(args.subjects_dir, subject_id, args.sessions_glob)
        epochs = []
        for idx, (label, stats_dir) in enumerate(sessions):
            if not stats_dir.exists():
                print(f"[{subject_id}] skipping missing session dir {stats_dir}")
                continue
            rois = load_hemispheres(stats_dir)
            graph = build_msn(rois, args.features, args.edge_threshold)
            epochs.append({"epoch_index": idx, "nodes": graph["nodes"], "edges": graph["edges"]})
            print(f"[{subject_id}] session {label}: {len(graph['nodes'])} ROIs, {len(graph['edges'])} edges")
        if not epochs:
            print(f"[{subject_id}] no usable sessions found, skipping")
            continue
        out_path = args.out_dir / f"{subject_id}.json"
        out_path.write_text(json.dumps({"subject_id": subject_id, "epochs": epochs}))
        print(f"[{subject_id}] wrote {len(epochs)} session(s) -> {out_path}")


if __name__ == "__main__":
    main()
