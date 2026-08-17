//! Aggregates a subject's belief trajectory (from `hmm_runtime`) and raw
//! per-epoch spectral features into the summary numbers written to the
//! screening report.

use neural_hmm::Belief;

use crate::spectral_features::{ASYMMETRY_IDX, SPECTRAL_GAP_IDX};
use crate::types::ScreeningRow;

fn entropy(p: &[f32]) -> f32 {
    -p.iter()
        .filter(|&&x| x > 1e-9)
        .map(|&x| x * x.ln())
        .sum::<f32>()
}

pub fn summarize(
    subject_id: &str,
    trajectory: &[Belief],
    raw_features: &[Vec<f32>],
    num_states: usize,
) -> ScreeningRow {
    let n = trajectory.len().max(1);

    let mut occupancy = vec![0f32; num_states];
    let mut entropy_sum = 0f32;
    let mut prev_argmax: Option<usize> = None;
    let mut switches = 0usize;

    for belief in trajectory {
        let slice = belief.as_slice();
        for (s, v) in slice.iter().enumerate() {
            occupancy[s] += v / n as f32;
        }
        entropy_sum += entropy(slice);
        let (argmax, _) = belief.argmax();
        if let Some(prev) = prev_argmax {
            if prev != argmax {
                switches += 1;
            }
        }
        prev_argmax = Some(argmax);
    }

    let switch_rate = if trajectory.len() > 1 {
        switches as f32 / (trajectory.len() - 1) as f32
    } else {
        0.0
    };
    let mean_entropy = entropy_sum / n as f32;

    let mean_asymmetry = mean_index(raw_features, ASYMMETRY_IDX);
    let mean_gap = mean_index(raw_features, SPECTRAL_GAP_IDX);

    ScreeningRow {
        subject_id: subject_id.to_string(),
        num_epochs: trajectory.len(),
        num_network_states: num_states,
        state_occupancy: occupancy,
        state_switch_rate: switch_rate,
        mean_belief_entropy: mean_entropy,
        mean_spectral_asymmetry: mean_asymmetry,
        mean_spectral_gap: mean_gap,
        // Placeholder; the CLI rescales this to [0, 1] across the cohort
        // after every subject has been summarized (see main.rs) since a
        // per-subject min-max wouldn't mean anything on its own.
        network_instability_index: switch_rate * mean_entropy,
    }
}

fn mean_index(raw_features: &[Vec<f32>], idx: usize) -> f32 {
    if raw_features.is_empty() {
        return 0.0;
    }
    raw_features.iter().filter_map(|f| f.get(idx)).sum::<f32>() / raw_features.len() as f32
}
