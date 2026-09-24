# Diffusion review follow-up

The runtime iteration keeps the eight-layer denoiser and 100-step categorical and Gaussian
processes. Validation uses the shared dataset protocol described in the README.

Two model questions remain open for a separate correctness investigation:

- `compute_loss` prohibits initial-position EOT in predicted clean probabilities at every
  timestep, while `generate` applies this constraint only at the final reverse step. Check the
  effect of using a consistent clean-state constraint at intermediate steps.
- The clean canvas fills positions after termination with EOT and standardized zero time, but
  activity supervision ends at the first EOT and time supervision ends before it. Bidirectional
  attention can read the unsupervised trailing positions. Check whether their generated states
  differ materially from training inputs and affect valid suffix positions.

These observations are hypotheses about sample quality, not established causes of poor results.
Evaluate any changes on train and validation data, with the shared validation protocol. Model
width, depth, noise schedules, loss balance, and sampling budgets remain subsequent ablations.
