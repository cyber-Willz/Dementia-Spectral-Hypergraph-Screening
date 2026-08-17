//! Turns one epoch's weighted connectivity graph into a fixed-length
//! spectral feature vector, via `spectral_hypergraph`'s normalized
//! Laplacian + dense eigen-decomposition.
//!
//! An ordinary weighted graph is represented as a hypergraph whose
//! hyperedges all have exactly 2 members -- this makes the normalized
//! hypergraph Laplacian of Zhou/Huang/Schoelkopf reduce exactly to the
//! standard normalized graph Laplacian, so we get `spectral_hypergraph`'s
//! validated builder, error handling, and eigensolver for free instead of
//! re-implementing graph Laplacians from scratch.

use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;
use spectral_hypergraph::spectral::dense_eigen;
use spectral_hypergraph::Result as HgResult;

use crate::types::EpochGraph;

/// Number of algebraically-smallest non-trivial eigenvalues retained.
const NUM_EIGVALS: usize = 6;

/// Feature vector layout: [lambda_1..lambda_NUM_EIGVALS, spectral_gap,
/// eigenvalue_entropy, fiedler_asymmetry] = NUM_EIGVALS + 3.
pub const FEATURE_DIM: usize = NUM_EIGVALS + 3;

/// Index into the feature vector of the frontal/posterior asymmetry term,
/// and of the spectral gap term -- exposed so the HMM-runtime layer can
/// pull them back out of the raw feature vector for the screening report
/// without recomputing anything.
pub const ASYMMETRY_IDX: usize = NUM_EIGVALS + 2;
pub const SPECTRAL_GAP_IDX: usize = NUM_EIGVALS;

/// Nodes whose label matches one of these (case-insensitive substring)
/// counts toward the "frontal" half of the asymmetry split; anything else
/// counts toward "posterior". This is deliberately coarse (works for both
/// 10-20 EEG channel names like `Fp1`/`F7` and FreeSurfer ROI names like
/// `superiorfrontal`/`parsopercularis`) -- it is a screening heuristic, not
/// an anatomical claim.
const FRONTAL_HINTS: [&str; 6] = ["f", "fp", "frontal", "orbito", "precentral", "pars"];

fn is_frontal(label: &str) -> bool {
    let lower = label.to_lowercase();
    // Channel-name case (Fp1, F3, F7, Fz, F4, F8): starts with 'f'.
    if lower.starts_with('f') {
        return true;
    }
    FRONTAL_HINTS.iter().any(|h| lower.contains(h))
}

/// Build the 2-uniform hypergraph (= ordinary weighted graph) for one
/// epoch and run the dense eigen-decomposition of its normalized
/// Laplacian.
fn laplacian_eigen(epoch: &EpochGraph) -> HgResult<(Vec<f64>, Vec<Vec<f64>>, Vec<bool>)> {
    let mut builder = HypergraphBuilder::with_capacity(epoch.nodes.len(), epoch.edges.len(), 2);
    for node in &epoch.nodes {
        builder.get_or_add_vertex(node.clone())?;
    }
    for edge in &epoch.edges {
        let a = builder.get_or_add_vertex(edge.source.clone())?;
        let b = builder.get_or_add_vertex(edge.target.clone())?;
        if a == b {
            continue; // no self-loops in the Laplacian
        }
        let w = edge.weight.max(1e-6); // Laplacian needs strictly positive weights
        builder.add_hyperedge(&[a, b], w)?;
    }
    let hg = builder.build()?;

    let laplacian = dense_normalized_laplacian(&hg)?;
    let eig = dense_eigen(&laplacian);

    let frontal_mask: Vec<bool> = epoch.nodes.iter().map(|n| is_frontal(n)).collect();
    let eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    let eigenvectors: Vec<Vec<f64>> = (0..eig.eigenvectors.ncols())
        .map(|c| eig.eigenvectors.column(c).iter().copied().collect())
        .collect();
    Ok((eigenvalues, eigenvectors, frontal_mask))
}

/// Extract the fixed-length spectral feature vector for one epoch.
///
/// Isolated nodes (zero-degree, e.g. a channel with no surviving edges
/// after thresholding) are dropped before building the hypergraph, since
/// `spectral_hypergraph` rejects isolated vertices (D_v^{-1/2} undefined).
pub fn extract(epoch: &EpochGraph) -> anyhow::Result<[f32; FEATURE_DIM]> {
    // Drop nodes with no incident edges.
    let connected: std::collections::HashSet<&str> = epoch
        .edges
        .iter()
        .flat_map(|e| [e.source.as_str(), e.target.as_str()])
        .collect();
    let filtered_nodes: Vec<String> = epoch
        .nodes
        .iter()
        .filter(|n| connected.contains(n.as_str()))
        .cloned()
        .collect();
    if filtered_nodes.len() < 3 {
        anyhow::bail!(
            "epoch {} has fewer than 3 connected nodes ({}); cannot compute a meaningful Laplacian spectrum",
            epoch.epoch_index,
            filtered_nodes.len()
        );
    }
    let filtered = EpochGraph {
        epoch_index: epoch.epoch_index,
        nodes: filtered_nodes,
        edges: epoch.edges.clone(),
    };

    let (eigenvalues, eigenvectors, frontal_mask) =
        laplacian_eigen(&filtered).map_err(|e| anyhow::anyhow!("laplacian/eigen failure: {e}"))?;

    let mut out = [0f32; FEATURE_DIM];

    // lambda_1..lambda_NUM_EIGVALS: skip the trivial lambda_0 (~0, constant
    // eigenvector), keep the next NUM_EIGVALS -- these encode how tightly
    // clustered / well-connected the epoch's network is.
    for i in 0..NUM_EIGVALS {
        out[i] = *eigenvalues.get(i + 1).unwrap_or(&0.0) as f32;
    }

    // Spectral gap: lambda_2 - lambda_1 (algebraic connectivity margin).
    let l1 = *eigenvalues.get(1).unwrap_or(&0.0);
    let l2 = *eigenvalues.get(2).unwrap_or(&l1);
    out[SPECTRAL_GAP_IDX] = (l2 - l1) as f32;

    // Eigenvalue entropy: treat the normalized non-trivial eigenvalues as a
    // distribution and take their Shannon entropy -- a scalar summary of
    // how spread out (vs. concentrated) the graph's spectrum is.
    let tail: Vec<f64> = eigenvalues.iter().skip(1).copied().collect();
    let sum: f64 = tail.iter().sum::<f64>().max(1e-9);
    let entropy: f64 = -tail
        .iter()
        .map(|v| {
            let p = (v / sum).max(1e-12);
            p * p.ln()
        })
        .sum::<f64>();
    out[NUM_EIGVALS + 1] = entropy as f32;

    // Frontal/posterior asymmetry: project the Fiedler vector (index 1)
    // onto the frontal vs. posterior node partition and compare mean
    // magnitudes.
    if let Some(fiedler) = eigenvectors.get(1) {
        let (mut frontal_sum, mut frontal_n, mut post_sum, mut post_n) = (0.0, 0usize, 0.0, 0usize);
        for (v, &is_f) in fiedler.iter().zip(frontal_mask.iter()) {
            if is_f {
                frontal_sum += v.abs();
                frontal_n += 1;
            } else {
                post_sum += v.abs();
                post_n += 1;
            }
        }
        let frontal_mean = if frontal_n > 0 { frontal_sum / frontal_n as f64 } else { 0.0 };
        let post_mean = if post_n > 0 { post_sum / post_n as f64 } else { 0.0 };
        out[ASYMMETRY_IDX] = (frontal_mean - post_mean) as f32;
    }

    Ok(out)
}
