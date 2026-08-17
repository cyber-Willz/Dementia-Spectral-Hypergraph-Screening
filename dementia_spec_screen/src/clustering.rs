//! Minimal k-means, used only to derive pseudo-labels for the K latent
//! "network states" the HMM operates over. This is unsupervised -- it never
//! sees group labels (AD/FTD/CN), only the pooled spectral feature vectors
//! across all subjects/epochs in the run. Keeping it label-blind avoids
//! leaking group information into the state definitions themselves; group
//! discrimination happens later, downstream, on the aggregated screening
//! features (see the Python `fit_eval.py`).

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Lloyd's algorithm with k-means++ initialization, fixed iteration budget.
/// `k` should be small (this crate treats it as the number of coarse
/// large-scale network states, typically 3-6) and `points` typically
/// numbers in the hundreds to low thousands of epoch feature vectors, so a
/// dependency-free implementation is plenty fast.
pub fn kmeans(points: &[Vec<f32>], k: usize, max_iter: usize, seed: u64) -> KMeansResult {
    assert!(!points.is_empty(), "kmeans requires at least one point");
    let k = k.min(points.len());
    let dim = points[0].len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // k-means++ initialization.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(points.choose(&mut rng).unwrap().clone());
    while centroids.len() < k {
        let weights: Vec<f32> = points
            .iter()
            .map(|p| centroids.iter().map(|c| dist2(p, c)).fold(f32::MAX, f32::min))
            .collect();
        let total: f32 = weights.iter().sum::<f32>().max(1e-9);
        let mut r = rng.gen::<f32>() * total;
        let mut chosen = points.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            if r <= w {
                chosen = i;
                break;
            }
            r -= w;
        }
        centroids.push(points[chosen].clone());
    }

    let mut assignments = vec![0usize; points.len()];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = dist2(p, c);
                if d < best_d {
                    best_d = d;
                    best = c_idx;
                }
            }
            if assignments[i] != best {
                changed = true;
            }
            assignments[i] = best;
        }
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (p, &a) in points.iter().zip(assignments.iter()) {
            counts[a] += 1;
            for d in 0..dim {
                sums[a][d] += p[d];
            }
        }
        for c_idx in 0..k {
            if counts[c_idx] == 0 {
                continue; // keep previous centroid for an empty cluster
            }
            for d in 0..dim {
                centroids[c_idx][d] = sums[c_idx][d] / counts[c_idx] as f32;
            }
        }
        if !changed {
            break;
        }
    }

    KMeansResult { centroids, assignments }
}

use rand::Rng;
