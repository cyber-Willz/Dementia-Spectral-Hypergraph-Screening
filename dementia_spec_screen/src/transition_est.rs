//! Empirical (MLE, Laplace-smoothed) transition matrix estimation from a
//! pooled set of pseudo-label sequences (one sequence per subject, in
//! epoch order). This feeds `neural_hmm::TransitionMatrix`, which only
//! requires a validated row-stochastic matrix -- how it was estimated is
//! out of its scope.

use neural_hmm::{HmmResult, TransitionMatrix};

/// `sequences[s]` is subject `s`'s ordered list of pseudo-labels (one per
/// epoch, values in `0..num_states`).
pub fn estimate(sequences: &[Vec<usize>], num_states: usize) -> HmmResult<TransitionMatrix> {
    let mut counts = vec![vec![1.0f32; num_states]; num_states]; // Laplace smoothing
    for seq in sequences {
        for pair in seq.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            counts[from][to] += 1.0;
        }
    }
    let rows: Vec<Vec<f32>> = counts
        .into_iter()
        .map(|row| {
            let sum: f32 = row.iter().sum();
            row.into_iter().map(|c| c / sum).collect()
        })
        .collect();
    TransitionMatrix::new(rows)
}
