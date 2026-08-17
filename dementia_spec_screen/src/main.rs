//! `screen`: CLI that runs the full dementia_spec_screen pipeline over a
//! directory of per-subject epoch-graph JSON files (see `types::SubjectRecord`)
//! and writes a per-subject screening report CSV.
//!
//! Usage:
//!   screen <input_dir> <output_csv> [num_states] [epochs] [batch_size] [lr]
//!
//! `input_dir` must contain one `*.json` file per subject, each parsing as
//! `types::SubjectRecord` (see `python/eeg_feature_extraction.py` or
//! `python/mri_feature_extraction.py` for how those are produced from raw
//! OpenNeuro/BIDS data).

use std::fs;
use std::path::Path;

use dementia_spec_screen::emission_train::train;
use dementia_spec_screen::hmm_runtime::run_subject;
use dementia_spec_screen::screening::summarize;
use dementia_spec_screen::spectral_features::{extract, FEATURE_DIM};
use dementia_spec_screen::transition_est::estimate;
use dementia_spec_screen::types::{ScreeningRow, SubjectRecord};
use dementia_spec_screen::clustering::kmeans;

use neural_hmm::NeuralHmm;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: screen <input_dir> <output_csv> [num_states=4] [epochs=200] [batch_size=32] [lr=0.01]"
        );
        std::process::exit(1);
    }
    let input_dir = &args[1];
    let output_csv = &args[2];
    let num_states: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let train_epochs: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(200);
    let batch_size: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let lr: f64 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(0.01);

    // --- 1. Load subjects and extract per-epoch spectral features ---
    let mut subjects: Vec<(String, Vec<[f32; FEATURE_DIM]>)> = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let record: SubjectRecord = serde_json::from_str(&fs::read_to_string(&path)?)?;
        let mut feats = Vec::with_capacity(record.epochs.len());
        for epoch in &record.epochs {
            match extract(epoch) {
                Ok(f) => feats.push(f),
                Err(e) => eprintln!(
                    "warning: subject {} epoch {} skipped: {e}",
                    record.subject_id, epoch.epoch_index
                ),
            }
        }
        if feats.is_empty() {
            eprintln!("warning: subject {} had no usable epochs, skipping", record.subject_id);
            continue;
        }
        subjects.push((record.subject_id, feats));
    }
    if subjects.is_empty() {
        anyhow::bail!("no usable subject files found in {input_dir}");
    }
    println!("loaded {} subjects", subjects.len());

    // --- 2. Pool epoch features across all subjects, run k-means ---
    let pooled: Vec<Vec<f32>> = subjects
        .iter()
        .flat_map(|(_, feats)| feats.iter().map(|f| f.to_vec()))
        .collect();
    let km = kmeans(&pooled, num_states, 100, 42);
    println!("k-means converged with {} states over {} pooled epochs", num_states, pooled.len());

    // Re-split pooled labels back out per-subject, in original order.
    let mut label_cursor = 0usize;
    let mut per_subject_labels: Vec<Vec<usize>> = Vec::with_capacity(subjects.len());
    for (_, feats) in &subjects {
        let n = feats.len();
        per_subject_labels.push(km.assignments[label_cursor..label_cursor + n].to_vec());
        label_cursor += n;
    }

    // --- 3. Estimate the transition matrix from pseudo-label sequences ---
    let transition = estimate(&per_subject_labels, num_states)?;

    // --- 4. Train the emission MLP against the pseudo-labels ---
    let trained = train(&pooled, &km.assignments, num_states, train_epochs, batch_size, lr, 7);
    println!("emission engine trained, final mean batch loss = {:.4}", trained.final_loss);

    let hmm = NeuralHmm::new(trained.engine, transition)?;

    // --- 5. Run the HMM forward per subject, summarize ---
    let mut rows: Vec<ScreeningRow> = Vec::with_capacity(subjects.len());
    for (subject_id, feats) in &subjects {
        let feats_vec: Vec<Vec<f32>> = feats.iter().map(|f| f.to_vec()).collect();
        let trajectory = run_subject(&hmm, &trained.device, &feats_vec)?;
        rows.push(summarize(subject_id, &trajectory, &feats_vec, num_states));
    }

    // --- 6. Cohort-normalize the heuristic instability index to [0, 1] ---
    let (min_i, max_i) = rows.iter().fold((f32::MAX, f32::MIN), |(mn, mx), r| {
        (mn.min(r.network_instability_index), mx.max(r.network_instability_index))
    });
    let span = (max_i - min_i).max(1e-9);
    for r in &mut rows {
        r.network_instability_index = (r.network_instability_index - min_i) / span;
    }

    write_csv(output_csv, &rows, num_states)?;
    println!("wrote {} rows to {output_csv}", rows.len());
    println!(
        "\nNOTE: this is an unsupervised, exploratory screening signal only -- it has not \
         been clinically validated. See README.md before drawing any conclusions from it."
    );
    Ok(())
}

fn write_csv(path: &str, rows: &[ScreeningRow], num_states: usize) -> anyhow::Result<()> {
    let mut out = String::new();
    out.push_str("subject_id,num_epochs,");
    for s in 0..num_states {
        out.push_str(&format!("state_occupancy_{s},"));
    }
    out.push_str(
        "state_switch_rate,mean_belief_entropy,mean_spectral_asymmetry,mean_spectral_gap,network_instability_index\n",
    );
    for r in rows {
        out.push_str(&format!("{},{},", r.subject_id, r.num_epochs));
        for v in &r.state_occupancy {
            out.push_str(&format!("{v:.6},"));
        }
        out.push_str(&format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            r.state_switch_rate,
            r.mean_belief_entropy,
            r.mean_spectral_asymmetry,
            r.mean_spectral_gap,
            r.network_instability_index
        ));
    }
    fs::write(Path::new(path), out)?;
    Ok(())
}
