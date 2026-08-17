//! Drives `neural_hmm::NeuralHmm::filter_step` across one subject's ordered
//! epoch feature sequence, producing a belief trajectory.

use burn::tensor::Tensor;
use neural_hmm::{Belief, HmmResult, NeuralHmm};

use crate::emission_train::TrainBackend;

/// One belief vector (posterior over the K network states) per epoch, in
/// epoch order.
pub fn run_subject(
    hmm: &NeuralHmm<TrainBackend>,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
    epoch_features: &[Vec<f32>],
) -> HmmResult<Vec<Belief>> {
    let mut belief = Belief::uniform(hmm.num_states())?;
    let mut trajectory = Vec::with_capacity(epoch_features.len());
    for feat in epoch_features {
        let input: Tensor<TrainBackend, 2> =
            Tensor::from_floats(feat.as_slice(), device).reshape([1, feat.len()]);
        belief = hmm.filter_step(&belief, input)?;
        trajectory.push(belief.clone());
    }
    Ok(trajectory)
}
