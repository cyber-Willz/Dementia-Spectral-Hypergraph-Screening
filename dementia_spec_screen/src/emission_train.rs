//! Fits `neural_hmm`'s `NeuralEmissionEngine` MLP to predict the k-means
//! pseudo-label (network state) of each pooled spectral feature vector.
//!
//! `neural_hmm` ships the emission model architecture but deliberately no
//! training loop (it's a production *inference* crate -- see its README:
//! "the transition matrix here is static/hand-specified", training is
//! flagged as a separate offline step). This module is that offline step,
//! specialized to this pipeline: cross-entropy against cluster pseudo-labels
//! rather than true class labels, since at this stage we're only trying to
//! get a differentiable, smoothed stand-in for "which network state is this
//! epoch closest to", not predicting diagnosis.

use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use neural_hmm::{EmissionEngineConfig, NeuralEmissionEngine};

pub type TrainBackend = Autodiff<NdArray<f32>>;

pub struct TrainedEmission {
    pub engine: NeuralEmissionEngine<TrainBackend>,
    pub device: <TrainBackend as burn::tensor::backend::Backend>::Device,
    pub final_loss: f32,
}

/// `features`: pooled epoch feature vectors, `labels[i]` is the k-means
/// cluster id for `features[i]`. Trains a small MLP for `epochs` passes
/// over the full (small) dataset with Adam, batching in chunks of
/// `batch_size`.
pub fn train(
    features: &[Vec<f32>],
    labels: &[usize],
    num_states: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    seed: u64,
) -> TrainedEmission {
    assert_eq!(features.len(), labels.len());
    let device = Default::default();
    let input_dim = features[0].len();
    let config = EmissionEngineConfig::new(input_dim, num_states);
    let mut model = NeuralEmissionEngine::<TrainBackend>::new(&device, &config);
    let mut optimizer = AdamConfig::new().init();

    let n = features.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::from_seed_u64(seed);

    let mut final_loss = 0f32;
    for _epoch in 0..epochs {
        shuffle(&mut order, &mut rng);
        let mut epoch_loss = 0f32;
        let mut num_batches = 0usize;
        for chunk in order.chunks(batch_size.max(1)) {
            let batch_dim = chunk.len();
            let mut flat = Vec::with_capacity(batch_dim * input_dim);
            for &idx in chunk {
                flat.extend(features[idx].iter().copied());
            }
            let input: Tensor<TrainBackend, 2> =
                Tensor::from_floats(flat.as_slice(), &device).reshape([batch_dim, input_dim]);
            let log_probs = model.forward_log_probs(input);

            let targets: Vec<i32> = chunk.iter().map(|&idx| labels[idx] as i32).collect();
            let target_tensor: Tensor<TrainBackend, 1, burn::tensor::Int> =
                Tensor::from_ints(targets.as_slice(), &device);
            let loss = nll_loss(log_probs, target_tensor, num_states);

            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grad_params);

            epoch_loss += loss.into_scalar();
            num_batches += 1;
        }
        final_loss = epoch_loss / (num_batches.max(1) as f32);
    }

    TrainedEmission { engine: model, device, final_loss }
}

/// Negative log-likelihood loss from `log_probs` ([batch, num_states]) and
/// integer `targets` ([batch]), implemented by hand (burn 0.13's
/// `NllLoss` module works, but gathering by index directly keeps this
/// module dependency-light and easy to audit).
fn nll_loss<B: AutodiffBackend>(
    log_probs: Tensor<B, 2>,
    targets: Tensor<B, 1, burn::tensor::Int>,
    num_states: usize,
) -> Tensor<B, 1> {
    let [batch, _] = log_probs.dims();
    let targets_2d = targets.reshape([batch, 1]);
    let gathered = log_probs.gather(1, targets_2d); // [batch, 1]
    let _ = num_states;
    -gathered.mean()
}

fn shuffle(order: &mut [usize], rng: &mut rand_chacha::ChaCha8Rng) {
    use rand::seq::SliceRandom;
    order.shuffle(rng);
}

trait FromSeedU64 {
    fn from_seed_u64(seed: u64) -> Self;
}
impl FromSeedU64 for rand_chacha::ChaCha8Rng {
    fn from_seed_u64(seed: u64) -> Self {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(seed)
    }
}
