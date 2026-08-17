//! Data schema shared with the Python feature-extraction side.
//!
//! A "subject file" is one JSON document per subject containing an ordered
//! sequence of epoch-level connectivity graphs (one per sliding window of
//! the recording -- EEG functional connectivity or an MRI-derived
//! morphometric similarity network; the Rust side is modality-agnostic, it
//! only ever sees weighted graphs over named nodes).

use serde::{Deserialize, Serialize};

/// One weighted edge in an epoch's connectivity graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub source: String,
    pub target: String,
    /// Connectivity strength for this epoch/band (e.g. weighted phase-lag
    /// index, coherence, or morphometric-similarity correlation). Must be
    /// finite and non-negative -- rectify/rescale upstream if your metric
    /// can go negative (e.g. shift-and-clip PLI/correlation to [0, 1]).
    pub weight: f64,
}

/// One epoch (sliding window) of connectivity for one subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochGraph {
    pub epoch_index: usize,
    /// Node labels in a fixed, subject-independent order (e.g. 10-20 EEG
    /// channel names, or FreeSurfer aparc ROI names). All epochs for a
    /// subject -- and ideally across subjects -- should use the same node
    /// set so spectral features are comparable.
    pub nodes: Vec<String>,
    pub edges: Vec<EdgeRecord>,
}

/// A full subject record: identifier plus its ordered epoch sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectRecord {
    pub subject_id: String,
    pub epochs: Vec<EpochGraph>,
}

/// Per-epoch spectral feature vector derived from the connectivity graph's
/// normalized Laplacian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub epoch_index: usize,
    /// Fixed-length feature vector -- see `spectral_features::FEATURE_DIM`
    /// and `spectral_features::extract` for the exact composition.
    pub features: Vec<f32>,
}

/// Final per-subject output row written to the screening report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningRow {
    pub subject_id: String,
    pub num_epochs: usize,
    pub num_network_states: usize,
    /// Mean soft occupancy of each latent network state across the
    /// recording (sums to ~1.0).
    pub state_occupancy: Vec<f32>,
    /// Fraction of consecutive epochs where the most-likely state changes.
    pub state_switch_rate: f32,
    /// Mean Shannon entropy (nats) of the belief distribution across
    /// epochs -- how "undecided" the filter stays about which network
    /// state is active.
    pub mean_belief_entropy: f32,
    /// Mean spectral-asymmetry feature (see spectral_features) across
    /// epochs -- a coarse frontal/posterior connectivity balance measure.
    pub mean_spectral_asymmetry: f32,
    /// Mean spectral gap (lambda_2 - lambda_1 of the epoch Laplacian)
    /// across epochs -- lower values indicate a more fragmented /
    /// weakly-integrated connectivity graph.
    pub mean_spectral_gap: f32,
    /// Heuristic composite screening score = state_switch_rate *
    /// mean_belief_entropy, min-max normalized against the cohort processed
    /// in this run. NOT a diagnostic score -- see crate-level docs.
    pub network_instability_index: f32,
}
