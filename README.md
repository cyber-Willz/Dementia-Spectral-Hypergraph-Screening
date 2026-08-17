# dementia_spec_screen

A research pipeline that turns routine EEG (and, where available, MRI)
recordings into an unsupervised **network-dynamics screening signal**, and
evaluates how well that signal separates dementia subtypes on a public
cohort. It's built by extending the existing `spec_engine` /
`sec_net_engine` ecosystem (`spectral_hypergraph` for graph-spectral
features, `neural_hmm` for temporal state filtering) into a new domain
rather than starting from scratch.

## Read this first

**This is a research pipeline for investigating candidate screening
signals. It is not a diagnostic device.** It has not been clinically
validated, has no regulatory clearance (FDA/CE/etc.), and must not be used
to make, support, or influence any individual's actual diagnosis or care.
Two things make that concrete, not just a disclaimer:

- The Rust pipeline (`dementia_spec_screen`) never sees group labels.
  Clustering, emission-network training, and transition estimation are all
  done on connectivity features alone -- only the final Python evaluation
  step (`fit_eval.py`) touches ground truth, and only to *measure*
  separability, not to feed it back into the model.
- Every number `fit_eval.py` prints is a leave-one-subject-out estimate on
  a single small public cohort with no external replication, no held-out
  validation set, and no clinical calibration. Treat it as "is this signal
  worth investigating further", not "does this test work".

## Architecture

```
EEG (.set/.edf/...)  --python/eeg_feature_extraction.py-->   \
                                                                subject_graphs/*.json
MRI (FreeSurfer stats) --python/mri_feature_extraction.py-->  /   (types::SubjectRecord)
                                                                       |
                                                                       v
                                              dementia_spec_screen (Rust, cargo run --bin screen)
                                                       |
             1. spectral_features.rs  -- per-epoch connectivity graph -> Laplacian
                                          eigen-decomposition (via spectral_hypergraph)
             2. clustering.rs         -- k-means over pooled epoch features
                                          -> pseudo-labels for K latent "network states"
             3. emission_train.rs     -- trains neural_hmm's emission MLP (burn/autodiff)
                                          against the pseudo-labels
             4. transition_est.rs     -- empirical MLE transition matrix over
                                          pseudo-label sequences -> neural_hmm::TransitionMatrix
             5. hmm_runtime.rs        -- neural_hmm::NeuralHmm::filter_step per subject
                                          -> belief trajectory
             6. screening.rs          -- aggregates trajectory -> per-subject screening row
                                                       |
                                                       v
                                          screening_report.csv
                                                       |
                                     python/fit_eval.py (joins with participants.tsv,
                                     LOSO logistic regression, AUROC per group pair)
```

The Rust side is modality-agnostic: it only ever sees a sequence of
weighted graphs over named nodes (`types::EpochGraph`). Swapping EEG
connectivity for an MRI morphometric-similarity network -- or mixing both
in one workspace, run as separate `screen` invocations and compared -- is
a matter of which Python extractor you run, not a code change downstream.

## Dataset

The primary target dataset is **OpenNeuro ds004504**: "A dataset of EEG
recordings from: Alzheimer's disease, Frontotemporal Dementia and Healthy
subjects" (Miltiadous et al., *Data* 2023, doi:10.3390/data8060095;
doi:10.18112/openneuro.ds004504.v1.0.2, CC0). 88 subjects, resting-state
eyes-closed EEG, 19-channel 10-20 montage, raw + ICA/ASR-preprocessed
derivatives included, `participants.tsv` with `Group` in `{A, F, C}` =
`{AD, FTD, CN}` and MMSE scores.

Fetch it (it's hosted on OpenNeuro's S3-backed storage, not a plain git
remote, so use one of):

```bash
python3 -m venv venv
source venv/bin/activate
pip install openneuro-py mne numpy scipy scikit-learn pandas

pip install openneuro-py

openneuro-py download --dataset ds004504 --target-dir ./ds004504

# or, with the AWS CLI:
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./ds004504
```

There is currently no openly, directly downloadable dataset pairing MRI
with the same AD/FTD/CN subtype labels the way ds004504 does for EEG
(OASIS-3 and ADNI have MRI + diagnosis but require a data use agreement
and are not fetchable over plain HTTP/git). `mri_feature_extraction.py` is
written to work against **any** BIDS + FreeSurfer `recon-all` output, so
it's ready to point at such a cohort once you have legitimate access,
rather than pretending to fetch one automatically.

## Running it end to end

```bash
# 1. EEG -> per-subject connectivity graphs
pip install mne numpy scipy scikit-learn
python python/eeg_feature_extraction.py \
    --bids-root ./ds004504 --derivatives \
    --out-dir ./subject_graphs \
    --band alpha --epoch-seconds 5.0 --connectivity wpli

# (optional, if/when you have MRI access) FreeSurfer -> morphometric similarity networks
python python/mri_feature_extraction.py \
    --subjects-dir ./freesurfer_output \
    --out-dir ./subject_graphs_mri

# 2. Rust: spectral features -> HMM -> screening report
cargo run --release --bin screen -- ./subject_graphs ./screening_report.csv

# 3. Evaluate against ground truth (LOSO AUROC per group pair)
python python/fit_eval.py \
    --screening-report ./screening_report.csv \
    --participants-tsv ./ds004504/participants.tsv
```

I validated this exact flow end to end in a sandbox against synthetic data
matching the real schema (I can't reach ds004504's actual S3-hosted files
from here, network egress is restricted to crates.io/pypi/npm/github) --
`eeg_feature_extraction.py`'s band-filter/wPLI/coherence math, the full
Rust pipeline (k-means -> MLP training -> HMM filtering -> CSV), and
`fit_eval.py`'s LOSO evaluation all ran and produced sane, non-degenerate
output. Real ds004504 data will need a first run on your machine to
confirm actual (rather than structural) correctness -- start with
`--subjects sub-001 sub-002 sub-003` on each script to iterate quickly
before running the full 88-subject cohort.

## Known limitations / next steps

- **19-channel EEG montage is coarse.** Spectral/connectivity features at
  this density are a reasonable screening feature space but won't localize
  fine-grained cortical differences the way source-localized or high-density
  EEG would.
- **K (number of network states) and the epoch window length are
  hyperparameters**, not fit values -- `fit_eval.py`'s AUROC will move
  around with them. Worth a small grid search (e.g. K in 3..6, window in
  2..10s) reported honestly as a hyperparameter sweep, not cherry-picked.
- **The composite `network_instability_index` is a hand-specified
  heuristic** (switch_rate * mean_entropy), not something the pipeline
  learned. `fit_eval.py` uses the full feature vector, which is the more
  defensible number; the index is there for quick eyeballing/sorting, not
  as the "real" score.
- **No test/train split at the cohort level** -- LOSO on 88 subjects from
  one site/device is a reasonable first check but is not a substitute for
  an external replication cohort before this signal means anything
  clinically.
- If you want to go further: Baum-Welch re-estimation of the transition
  matrix instead of the current empirical MLE, a proper train/val split
  for the emission MLP's hyperparameters, and swapping wPLI for
  source-localized connectivity are the highest-leverage next steps.
