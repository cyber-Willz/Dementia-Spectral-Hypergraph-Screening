//! # dementia_spec_screen
//!
//! Research pipeline that turns a sequence of per-epoch brain connectivity
//! graphs (EEG functional connectivity, or an MRI-derived morphometric
//! similarity network) into a per-subject **network-dynamics screening
//! signal**, by combining:
//!
//! 1. [`spectral_features`] -- per-epoch graph spectral features via
//!    `spectral_hypergraph`'s normalized Laplacian + eigen-decomposition.
//! 2. [`clustering`] -- unsupervised k-means over pooled epoch features to
//!    define a small number of latent "network states".
//! 3. [`emission_train`] -- fits `neural_hmm`'s emission MLP to those
//!    pseudo-labels.
//! 4. [`transition_est`] -- empirical Markov transition matrix over the
//!    pseudo-label sequences.
//! 5. [`hmm_runtime`] -- runs `neural_hmm::NeuralHmm::filter_step` across
//!    each subject's epoch sequence to get a belief trajectory.
//! 6. [`screening`] -- aggregates each trajectory into a fixed-length
//!    per-subject screening feature row.
//!
//! ## What this is and is not
//!
//! This crate never sees group labels (AD / FTD / CN, or any other
//! diagnosis) -- clustering, emission training, and transition estimation
//! are all done on connectivity features alone. The screening features it
//! outputs are an *unsupervised, exploratory summary of network dynamics*.
//! Whether -- and how well -- those features separate diagnostic groups is
//! an empirical question answered downstream (see the companion
//! `fit_eval.py`, which does a leave-one-subject-out evaluation against
//! ground-truth labels).
//!
//! **This is a research pipeline for investigating candidate screening
//! signals, not a diagnostic device.** It has not been clinically
//! validated, is not regulatory-cleared (e.g. FDA/CE), and must not be
//! used to make, support, or influence an individual patient's diagnosis
//! or care. See the top-level README for details.

pub mod clustering;
pub mod emission_train;
pub mod hmm_runtime;
pub mod screening;
pub mod spectral_features;
pub mod transition_est;
pub mod types;
