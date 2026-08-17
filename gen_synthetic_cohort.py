#!/usr/bin/env python3
"""
gen_synthetic_cohort.py
========================

Generates a synthetic cohort matching the exact schema
`eeg_feature_extraction.py` would produce from real ds004504 data (19-channel
10-20 montage, per-epoch weighted connectivity graphs), plus a matching
`participants.tsv`. This is NOT real EEG data and is not meant to represent
one -- it exists purely to drive a live, end-to-end run of the pipeline
(feature extraction schema -> Rust screen -> fit_eval) since ds004504 itself
lives on OpenNeuro's S3-backed storage, which this sandbox's network egress
allowlist does not include.

The synthetic generative model gives each group (AD, FTD, CN) a distinct
mean connectivity structure and a distinct latent-state switching rate, so
the pipeline has *something* structured to find -- it is explicitly not a
claim about what real AD/FTD/CN connectivity differences look like.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
N = len(CHANNELS)

# Group-level generative parameters (synthetic, not clinical claims):
#   base_weight: overall connectivity strength
#   frontal_boost: extra weight among frontal channels (F*, Fp*)
#   switch_prob: probability the latent network state flips each epoch
#   noise: per-edge Gaussian noise std
GROUP_PARAMS = {
    "C": dict(base_weight=0.55, frontal_boost=0.00, switch_prob=0.12, noise=0.06),  # control
    "A": dict(base_weight=0.42, frontal_boost=-0.10, switch_prob=0.30, noise=0.09),  # AD: weaker, more state-switchy
    "F": dict(base_weight=0.48, frontal_boost=-0.18, switch_prob=0.22, noise=0.08),  # FTD: frontal-specific drop
}

FRONTAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}

N_SUBJECTS_PER_GROUP = 10
N_EPOCHS = 24
SEED = 20260816


def make_epoch(rng: random.Random, params: dict, state: int) -> list[list[float]]:
    """Weighted, symmetric, non-negative connectivity matrix for one epoch."""
    w = [[0.0] * N for _ in range(N)]
    # state 0 = "engaged" (closer to base), state 1 = "disengaged" (globally weaker)
    state_scale = 1.0 if state == 0 else 0.75
    for i in range(N):
        for j in range(i + 1, N):
            base = params["base_weight"] * state_scale
            if CHANNELS[i] in FRONTAL and CHANNELS[j] in FRONTAL:
                base += params["frontal_boost"]
            val = base + rng.gauss(0.0, params["noise"])
            val = max(0.0, min(1.0, val))
            w[i][j] = val
            w[j][i] = val
    return w


def make_subject(subject_id: str, group: str, rng: random.Random) -> dict:
    params = GROUP_PARAMS[group]
    epochs = []
    state = 0
    for e in range(N_EPOCHS):
        if rng.random() < params["switch_prob"]:
            state = 1 - state
        mat = make_epoch(rng, params, state)
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append({"source": CHANNELS[i], "target": CHANNELS[j], "weight": mat[i][j]})
        epochs.append({"epoch_index": e, "nodes": list(CHANNELS), "edges": edges})
    return {"subject_id": subject_id, "epochs": epochs}


def main():
    out_dir = Path("subject_graphs")
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    rows = []
    sub_n = 1
    groups = ["A", "F", "C"]
    for group in groups:
        for _ in range(N_SUBJECTS_PER_GROUP):
            subject_id = f"sub-{sub_n:03d}"
            record = make_subject(subject_id, group, random.Random(SEED + sub_n))
            (out_dir / f"{subject_id}.json").write_text(json.dumps(record))
            age = rng.randint(56, 86)
            mmse = {"A": rng.randint(14, 24), "F": rng.randint(15, 26), "C": rng.randint(26, 30)}[group]
            gender = rng.choice(["M", "F"])
            rows.append(
                {"participant_id": subject_id, "Gender": gender, "Age": age, "Group": group, "MMSE": mmse}
            )
            sub_n += 1

    with open("participants.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "Gender", "Age", "Group", "MMSE"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {sub_n - 1} synthetic subjects to {out_dir}/ and participants.tsv")


if __name__ == "__main__":
    main()
