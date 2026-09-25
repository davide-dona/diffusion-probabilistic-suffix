# Masked Diffusion Transformer

This model predicts suffix content length from a prefix CLS summary. The denoiser then processes
the predicted number of positions with bidirectional suffix attention, prefix cross-attention, and
length and timestep conditioning. It shares event embeddings between prefix and suffix paths.

Training samples one diffusion step per trace. Activities at real suffix positions are replaced by
MASK with probability `1 - alpha_t`; at least one activity is masked per trace. Gaussian noise is
added to standardized inter-event times at real positions. The loss sums length cross entropy,
cross entropy averaged over masked activities per trace, and noise MSE averaged over real times
per trace. All three terms have configurable weights. The shared optimizer and validation regime
remain in `config/training/default.yaml`.

Generation reads the prefix only. It samples a valid content length, starts activities at MASK and
times at Gaussian noise, then applies the configured reverse timesteps. PAD, SOS, and EOT cannot be
emitted as content activities. Padded positions are cleared after denoising. Remaining time is
derived from the generated inter-event times using the fitted codecs.

For verification, use disposable CPU checks under `tmp/` and remove them afterward. Check finite
losses and gradients, length bounds, reverse reveal completion, prefix-only generation, and
derived remaining time. Do not run training or repository tests locally.
