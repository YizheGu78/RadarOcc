# Optional 3.7 m power-zeroing training

The original `RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py`
remains the full-input experiment. Its omitted `power_mask` defaults to disabled.
New config `RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1_mask37.py`
inherits it and enables:

```python
model = dict(temporal_cfg=dict(power_mask=dict(enabled=True, radius_m=3.7)))
```

The mask uses `hypot(raw_radar_x, raw_radar_y) <= 3.7` metres, matching the
audited XY circles, before ego alignment, at the existing P1 interface. It
applies to **all temporal frames including the current frame**. Channels 0,1,2
(selected powers) and 6,7 (mean, variance) become zero in a cloned tensor;
channels 3:6 (Doppler bin IDs), coordinates and point count remain unchanged.
GT and stored NPZs are not changed. Top-K membership is not reranked after
masking. This is power zeroing, not deleting sparse coordinates; the downstream
encoder can still use coordinates and Doppler IDs at those locations.

Zeroed historical points are excluded before local-neighbor search, so they
cannot occupy K/V slots or contribute projection-bias/position values. Current
masked points get no temporal correction and their power remains zero even
after the attention projection learns. The disabled path retains full candidates
and all raw descriptors. New mask settings introduce no parameters and do not
change checkpoint tensor names/shapes.

## Train from scratch

```bash
cd ~/projects/RadarOcc
git pull
source use_radarocc5060.sh

bash tools/dist_train.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1_mask37.py \
  1
```

Default: no loaded checkpoint, 15 epochs, normal epoch val. Separate directory:
`work_dirs/radarocc_small_temporal_t4_ego_attention_p1_mask37`.
Keep the original full-input experiment's checkpoints for the comparison.

## Test with the same mask

Choose the best checkpoint by val, then run (replace epoch number):

```bash
bash tools/dist_test.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1_mask37.py \
  work_dirs/radarocc_small_temporal_t4_ego_attention_p1_mask37/epoch_3.pth \
  1
```

Use this masked config for prediction export too (`--show --show-dir ...`).
The existing `run_temporal_t4_ego_attention_p1.sh` remains the full-input launcher;
do not use it to test a mask37 checkpoint. A CLI override
`--cfg-options model.temporal_cfg.power_mask.enabled=False` disables zeroing,
but inference-only disabling is a sensitivity check, not a separately trained
full-input baseline. For full-input training use the original config/work_dir.

This optional experiment combines power zeroing and excluding those points from
temporal neighbors. Compare it to the completed full-input training, with
otherwise matching data, optimizer, epochs and val selection rule.

## Finite log encoding for zero power

Small's actual MultiLayerRangeAttention applies log10 to the five continuous
power/statistic channels. The masked config therefore also sets
`pts_middle_encoder.allow_zero_power=True`: exact raw zeros encode as 0 after
log conversion, while strictly positive values retain their original log10
values. The logarithm receives 1 in zero slots before those encoded slots are
set to 0, so neither forward nor backward evaluates log10(0). Negative or
nonfinite power is rejected rather than silently repaired. Original configs
keep allow_zero_power=False and the original log path; checkpoint shapes and
parameter names remain unchanged. Raw power is still zero inside the region,
not replaced by epsilon in the fusion input.

Regression tests run the actual range-attention classes on CPU: reproduce
nonfinite outputs on zero inputs in the old path, verify finite forward/backward
in the new path, and exact matching on strictly positive inputs. CUDA sparse
convolution and full occupancy training still require the radarocc5060 machine.
