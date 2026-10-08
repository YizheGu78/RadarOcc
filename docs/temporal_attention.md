# Ego-aligned local attention at P1

Configuration: `RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py`.
The identity and original Small configurations retain their behavior.

Current P1 descriptors supply queries. Each historical frame supplies up to 8
nearest keys/values within 2 m after official-bin XYZ conversion and ego-pose
alignment. Relative XYZ / 2 m and relative frame offset / 3 feed a position/time
MLP. Time is a **frame gap**, not seconds; optional timestamps are not used.
Self-vehicle echoes are retained, including range-zero candidates.

The P1 input is 8 raw descriptor channels, not the sparse encoder's 11-D input.
The attention branch uses signed log1p and LayerNorm to handle the raw dynamic
range. Its 32-D, 4-head output predicts five bounded log-scale corrections for
channels 0,1,2,6,7. Fusion is the relative residual
`fused = current + current * expm1(0.5 * tanh(correction))`.
Channels 3:6 (Doppler bin IDs) and current sparse indices are unchanged.
Power remains positive and zero remains zero; this version cannot create a
return at an absent current anchor. It is P1 descriptor fusion, not fusion of
separately encoded historical sparse feature grids.

The final projection is zero initialized, so loading the validated single-frame
checkpoint reproduces current descriptors exactly. The projection learns first;
Q/K/V and position parameters acquire gradients once it becomes nonzero.
Invalid history is excluded from neighbor search. Fully empty neighborhoods
have zero residual, avoiding all-masked softmax NaNs. No-history batches are
supported by `find_unused_parameters=True` under DDP.

Neighbor search uses SciPy cKDTree on CPU; attention runs on the input device.
Queries are processed in chunks of 1024. This bounds attention matrices but
training still retains chunk activations for backward; measure actual GPU memory
on the 5060 Ti. No dense current-by-history distance matrix is allocated.

## Run

From the repository root, after `git pull`:

```bash
python -m unittest discover -s tests -p 'test_temporal*.py'

# First compare this initialization-only val result to the passed identity val.
bash run_temporal_t4_ego_attention_p1.sh val \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth

# Fresh optimizer/schedule, initialized from validated single-frame weights.
bash run_temporal_t4_ego_attention_p1.sh train \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth

# Evaluate a trained temporal checkpoint, not the initialization checkpoint.
bash run_temporal_t4_ego_attention_p1.sh test \
  work_dirs/radarocc_small_temporal_t4_ego_attention_p1/epoch_15.pth
```

MAT defaults to `data/K-Radar-official-meta/resources/info_arr.mat`.
The training configuration inherits the validated temporal train/val/test
pipelines and preserves Small's geometry queries, GT, losses and optimizer.
Checkpoint loading will report missing **new temporal attention parameters**;
this is expected for the old single-frame checkpoint. Other missing or mismatched
baseline weights require investigation before training.

Local tests cover exact identity, gradient flow after initialization, invalid
frames, radius rejection, T=1, Doppler ID preservation, and pose-dependent matching.
End-to-end MMCV/spconv inference, GPU memory and real-data IoU must be verified
in the radarocc5060 environment. This change does not implement temporal averaging
or establish that attention improves occupancy accuracy.
