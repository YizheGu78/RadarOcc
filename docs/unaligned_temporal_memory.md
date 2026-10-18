# Two-layer unaligned temporal memory at P1

This independent experiment uses four frames, original per-frame spherical
coordinates and relative frame gaps. It does not consume ego poses, warp
features, perform KNN matching, or call either existing temporal fusion method.
All pre-existing source files/configurations/scripts are unchanged. Selecting
an old config continues to register and execute the original model. New classes
are imported only through `plugin_dir='projects/unaligned_temporal/'`.

## Model

The existing Top-K preparation and P1 adapter yield `[B,T,N,8]`. A shared
signed-log descriptor MLP, a local spherical-index position MLP and a frame-gap
MLP map each return to 64 dimensions. No bin table is required. Coordinates are
NOT transformed to a common frame. The coordinate normalization constants are
scales, not physical calibration or an alignment operation.

For each valid history frame, 256 learned latent queries compress that frame
using cross-attention and an FFN. The compressor weights/queries are shared
across history frames. Current queries read the concatenated history memories
through TWO separately parameterized cross-attention + FFN blocks (4 heads,
FFN width 128). Each block has pre-norm and residual connections. There is no
quadratic full point self-attention. The memory-compression block is additional
to the two current-query fusion blocks. `num_layers=1/2/4` is configurable.

A gate and zero-initialized output projection generate five bounded log-power
corrections at the current sparse anchors. Channels 0,1,2,6,7 use
`current + current * expm1(0.5 * tanh(delta))`; Doppler bin IDs 3:6 are preserved
exactly. Initial predictions therefore match the corresponding single-frame
checkpoint, provided backbone weights/data settings match. The output learns
first; other temporal parameters acquire gradients after it becomes nonzero.
This P1 experiment cannot create an absent anchor or a nonzero return from a
zero-power anchor. It learns descriptor fusion, not a guaranteed physical motion
estimate. Memory compression may lose small-object detail; 256 is an initial
hyperparameter, not a demonstrated optimum.

The new dataset reuses the existing sequence/payload loader, requires scene ID
and order but no pose field, and computes actual order differences for frame
encoding. Identity pose placeholders satisfy the inherited payload schema only;
the new model never reads them. It keeps sequences within each scene and each
annotation split. Missing history is masked and only current GT is loaded.
Do not mix train and validation histories. Irregular subsets use actual order
spacing, not a claimed fixed 0.1 s interval.

## Run from repository root

The wrapper activates the existing `radarocc5060` environment just like the old
wrapper. Default training is from scratch for comparability with the current
KNN/RA-ADTA configs (same inherited 15 epochs/optimizer/loss/data pipelines).

```bash
git pull origin radarocc-5060-reproduction

# Recommended initial comparison: existing 20% train and validation annotations.
SUBSET=20pct bash run_temporal_t4_unaligned_memory_p1.sh train

# Validate this new checkpoint on the same 20% validation annotation.
SUBSET=20pct bash run_temporal_t4_unaligned_memory_p1.sh val \
  work_dirs/radarocc_small_temporal_t4_unaligned_memory_p1_20pct/epoch_15.pth

# Official test remains the FULL official test split, even with SUBSET=20pct.
SUBSET=20pct bash run_temporal_t4_unaligned_memory_p1.sh test \
  work_dirs/radarocc_small_temporal_t4_unaligned_memory_p1_20pct/epoch_15.pth

# Full-data training uses a separate work directory.
bash run_temporal_t4_unaligned_memory_p1.sh train

# Optional fine-tuning/initialization regression: use a SINGLE-FRAME checkpoint,
# not a checkpoint containing the old temporal_plugin parameters.
SUBSET=20pct bash run_temporal_t4_unaligned_memory_p1.sh train \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth

# Resume interrupted new-model training, preserving optimizer/epoch.
bash tools/dist_train.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_unaligned_memory_p1_20pct.py \
  1 --resume-from work_dirs/radarocc_small_temporal_t4_unaligned_memory_p1_20pct/latest.pth
```

To evaluate the initial identity against a matching single-frame baseline, pass
that single-frame checkpoint to the new `val` command. Missing new temporal
weights are expected; baseline weight shape mismatches are not. A trained old
KNN/RA-ADTA checkpoint is NOT interchangeable with the new temporal checkpoint.
Old checkpoints remain usable with their original configs/commands.

Direct configuration overrides are available through the existing entry points:
`--cfg-options model.unaligned_temporal.num_layers=4` (or 1),
`model.unaligned_temporal.memory_tokens=128`, and
`model.unaligned_temporal.chunk_size=512`. Use separate work directories for
architecture changes. `T` is set through `data.train/val/test.frame_nums`.

## Validation and limits

```bash
python -m unittest discover -s tests -p 'test_unaligned*.py' -v
python -m unittest discover -s tests -p 'test_temporal_attention.py' -v
python -m unittest discover -s tests -p 'test_ra_adta.py' -v
```

CPU tests cover identity, both fusion layers/compressor gradient flow, actual
history/time/position dependence, pose independence, invalid history including
NaN placeholders, T=1, bin ID preservation, chunk equivalence, high power,
sequence boundaries and pose-free metadata. Config inheritance is checked with
MMCV separately. CUDA/spconv end-to-end training, memory/FPS and IoU require the
real data and radarocc5060 environment and have not been measured here.

Chunking limits per-operation attention storage, but training retains activations
for backward; it does not guarantee a particular GPU memory usage. Learned memory
attention still costs O(T*N*M). Use `chunk_size`/`memory_tokens` overrides if
needed. The new config enables non-reentrant FPN checkpointing and unused-
parameter DDP support for no-history scene starts. It disables only the inherited
single-frame startup efficiency hook, which cannot accept temporal inputs.
