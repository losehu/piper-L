# Vendored realtime-vla

Source: https://github.com/dexmal/realtime-vla
Commit: `eb46c8d8ccf5216fd4bb71a48429f2da9dcf36a0` (MIT; see LICENSE).

This contains the Pi0 common kernels, Pi05 kernels and Pi05 weight converter.
Local changes in `pi05_infer.py`:

- Use a package-relative import for the common kernels.
- Optionally retain a copy of the two-view vision prefix immediately after its
  final projection, before language transformer layers overwrite the buffer.
- Allow a decoder step range, for separate CUDA graphs around OGG corrections.
- Cast the FP32 diffusion accumulator at the action-input projection.
- Allow the adapter to provide the final action update. The adapter retains the
  original unscaled output projection, the BF16 velocity/delta product, and an
  FP32 diffusion accumulator so small OGG corrections survive subsequent steps.

The converter adds a CPU tensor helper for Orbax `ml_dtypes.bfloat16` leaves.
The upstream CUDA-only CLI is not executed by this project; the CPU export reads
its static shape table, uses its weight layout conversion, and writes a separate
safetensors asset. The original checkpoint remains read-only.

The adapter uses the existing OpenPI tokenizer and data transforms, not the
upstream Hugging Face prompt path. The masked third camera is compacted out;
the two valid views and text retain their OpenPI logical RoPE positions. The
special optimized no-prompt PDL path is disabled for this state-token policy.
