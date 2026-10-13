#!/usr/bin/env python3
"""Record real RadarOcc Small Cartesian cross-attention samples for ONE frame.

Use from the repository root:

python tools/temporal_vis/record_cross_attention_sampling.py \
    projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py \
    work_dirs/radarocc_small_temporal_t4_ego_attention_p1/epoch_15.pth \
    --split val --index 0 --queries 256

This is inference-only. It does not modify model files, weights or the dataset.
Hooks collect small slices of the *actual* sampling-offset and attention-logit
layer outputs, then reconstruct CUDA sampling positions using the same formulas
as DeformCrossAttention3DCustom. Reference/offset values are not assumptions;
"target" centres are NOMINAL geometric centres, not learned feature locations.
"""
import argparse
import copy
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A locally trained legacy checkpoint is trusted by the owner. MMCV's older
# torch.load calls require this setting under PyTorch >= 2.6.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

OFFICIAL_RANGE_STEP_M = 0.462890625
SMALL_RANGE_STEP_M = 0.46 * 2.0


def summarize(v):
    """Finite-only descriptive statistics (numbers are ordinary floats)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"count": 0}
    return {
        "count": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p50": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def analyze_samples(reference, offsets, logits, spatial_shapes):
    """Analyze sampled tensors; axis order is [azimuth, range, elevation].

    reference: [N, L, 3] normalized coordinates recorded from the real forward
    offsets:   [N, heads, L, points, 3] actual linear-layer outputs
    logits:    [N, heads, L, points] actual attention logits
    spatial_shapes: [L, 3] axes [H, W, Z] (Small uses equal H and W)
    """
    reference = np.asarray(reference, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    spatial_shapes = np.asarray(spatial_shapes, dtype=np.float64)
    if spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 3:
        raise ValueError("Expected spatial_shapes [L,3]")
    n, levels, coords = reference.shape
    if coords != 3 or offsets.shape[:1] != (n,) or offsets.shape[2] != levels:
        raise ValueError("Reference / offset shape mismatch")
    if offsets.shape[-1] != 3 or offsets.shape[:-1] != logits.shape:
        raise ValueError("Offset / logit shape mismatch")
    if spatial_shapes.shape[0] != levels:
        raise ValueError("Spatial level count mismatch")

    # Exactly the order used by the CrossAttention source's offset_normalizer.
    norm = spatial_shapes[:, [1, 0, 2]]
    # CUDA ms_deform_attn_cuda_kernel.cuh: i = loc * size - 0.5.
    base = reference * norm[None, :, :] - 0.5
    actual = base[:, None, :, None, :] + offsets
    normalized_locations = reference[:, None, :, None, :] + (
        offsets / norm[None, None, :, None, :]
    )

    weights = np.exp(logits - logits.max(axis=(2, 3), keepdims=True))
    weights /= weights.sum(axis=(2, 3), keepdims=True)
    # Per-head weighted centroid across levels/points, averaged across heads.
    per_head_centroid = (actual * weights[..., None]).sum(axis=(2, 3))
    mean_sample = per_head_centroid.mean(axis=1)

    # Small has only one spherical level, with 128x128x14 resolution.
    # Nominal feature centre implied by padded indices [2r, 2a+149, 2e-9]
    # and two k3/p1/s2 sparse convolutions.
    if levels != 1 or not np.all(norm[0] == [128, 128, 14]):
        raise ValueError("Geometric target is only defined for Small [128,128,14]")
    base_one = base[:, 0, :]
    radar_range_m = reference[:, 0, 1] * 128.0 * SMALL_RANGE_STEP_M
    target = base_one.copy()
    target[:, 0] += 0.25  # Azimuth feature-index correction, nominal only.
    target[:, 1] = radar_range_m / (2.0 * OFFICIAL_RANGE_STEP_M)
    target[:, 2] += 0.25  # Elevation feature-index correction, nominal only.
    nominal_correction = target - base_one

    valid = np.isfinite(reference).all(axis=(1, 2)) & (
        (reference >= 0.0) & (reference <= 1.0)
    ).all(axis=(1, 2))
    valid &= (radar_range_m > 0.0) & (radar_range_m <= 80.55)
    # Real sampling points can lie outside, where CUDA interpolation is zero.
    inside_cuda = ((actual > -1.0) &
                   (actual < norm[None, None, :, None, :])).all(axis=-1)
    inside_centres = ((actual >= 0.0) &
                      (actual <= norm[None, None, :, None, :] - 1.0)).all(axis=-1)
    # Weighted sample position is a descriptive proxy, NOT the effective
    # feature position (attention fuses nonlinearly transformed feature values).
    weighted_correction = mean_sample - base_one
    weighted_residual = mean_sample - target
    nearest_euclidean = np.linalg.norm(actual[:, :, 0, :, :] - target[:, None, None, :], axis=-1).min(axis=(1, 2))

    metrics = {
        "sampled_queries": int(n),
        "valid_nominal_geometry_queries": int(valid.sum()),
        "attention_heads": int(offsets.shape[1]),
        "sampling_points_per_head": int(offsets.shape[3]),
        "offset_feature_indices": {
            axis: summarize(offsets[..., k][valid])
            for k, axis in enumerate(("azimuth", "range", "elevation"))
        },
        "nominal_reference_correction_feature_indices": {
            axis: summarize(nominal_correction[valid, k])
            for k, axis in enumerate(("azimuth", "range", "elevation"))
        },
        "attention_weighted_centroid_shift_feature_indices": {
            axis: summarize(weighted_correction[valid, k])
            for k, axis in enumerate(("azimuth", "range", "elevation"))
        },
        "weighted_centroid_minus_nominal_target_feature_indices": {
            axis: summarize(weighted_residual[valid, k])
            for k, axis in enumerate(("azimuth", "range", "elevation"))
        },
        "weighted_centroid_minus_target_degrees": {
            "azimuth": summarize(weighted_residual[valid, 0] * 2.0),
            "elevation": summarize(weighted_residual[valid, 2] * 2.0),
        },
        "nearest_sample_to_nominal_target_feature_indices": summarize(
            nearest_euclidean[valid]
        ),
        "sample_in_cuda_bounds_fraction": float(inside_cuda.mean()),
        "sample_in_feature_centres_fraction": float(inside_centres.mean()),
        "sampled_query_ranges_m": summarize(radar_range_m[valid]),
    }
    arrays = {
        "reference_normalized": reference.astype(np.float32),
        "base_feature": base.astype(np.float32),
        "offsets_feature_indices": offsets.astype(np.float32),
        "sampling_locations_normalized": normalized_locations.astype(np.float32),
        "sampling_feature_indices": actual.astype(np.float32),
        "attention_weights": weights.astype(np.float32),
        "nominal_target_feature": target.astype(np.float32),
        "nominal_correction_feature_indices": nominal_correction.astype(np.float32),
        "weighted_centroid_feature": mean_sample.astype(np.float32),
        "valid_nominal_geometry": valid,
        "radar_range_m": radar_range_m.astype(np.float32),
    }
    return metrics, arrays


class CrossAttentionCapture:
    def __init__(self, max_queries):
        if max_queries < 1:
            raise ValueError("--queries must be >= 1")
        self.max_queries = max_queries
        self.handles = []
        self.frames = {}
        self.modules = []

    def attach(self, model):
        from projects.occ_plugin.occupancy.deformable_transformer.deformable_cross_attention_3D_custom import (
            DeformCrossAttention3DCustom,
        )

        for name, mod in model.named_modules():
            if not name.startswith("transformer.") or not isinstance(mod, DeformCrossAttention3DCustom):
                continue
            if mod.num_bev_queue != 1:
                raise ValueError(f"{name}: expected num_bev_queue=1")
            self.modules.append(name)

            def before(module, args, kwargs, key=name):
                # The VoxFormerLayer3D passes these as kwargs.
                bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
                ref = bound.arguments.get("reference_points")
                shape = bound.arguments.get("spatial_shapes")
                if ref is None or shape is None:
                    raise RuntimeError(f"{key}: reference_points/spatial_shapes missing")
                if ref.ndim != 4 or ref.shape[-1] != 3:
                    raise RuntimeError(f"{key}: unexpected reference_points {tuple(ref.shape)}")
                valid = torch.isfinite(ref[0]).all(dim=(1, 2))
                valid &= ((ref[0] >= 0) & (ref[0] <= 1)).all(dim=(1, 2))
                ids = torch.nonzero(valid, as_tuple=False).flatten()
                if ids.numel() == 0:
                    raise RuntimeError(f"{key}: no valid reference points")
                keep = min(self.max_queries, ids.numel())
                chosen = ids[torch.linspace(0, ids.numel()-1, keep,
                                            device=ids.device).long()]
                ref_sample = ref[0].index_select(0, chosen).detach().float().cpu()
                spatial = torch.as_tensor(shape, dtype=torch.long).reshape(-1, 3)
                self.frames[key] = {
                    "query_indices": chosen,
                    "reference": ref_sample,
                    "spatial_shapes": spatial.cpu(),
                    "reference_batch": int(ref.shape[0]),
                }

            def offset_hook(module, args, output, key=name):
                self.frames[key]["offsets"] = output[0].index_select(
                    0, self.frames[key]["query_indices"]
                ).detach().float().cpu()

            def weight_hook(module, args, output, key=name):
                self.frames[key]["logits"] = output[0].index_select(
                    0, self.frames[key]["query_indices"]
                ).detach().float().cpu()

            self.handles.append(mod.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(mod.sampling_offsets.register_forward_hook(offset_hook))
            self.handles.append(mod.attention_weights.register_forward_hook(weight_hook))
        if not self.modules:
            raise RuntimeError("No transformer.* DeformCrossAttention3DCustom modules found")

    def save(self, out_dir, cfg_path, ckpt_path, split, dataset_index):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "scope": "Real inference: sampled cross-attention linear outputs; nominal-centre comparison only",
            "config": str(cfg_path),
            "checkpoint": str(ckpt_path),
            "split": split,
            "dataset_index": dataset_index,
            "modules": {},
            "limitations": [
                "Only a deterministic subset of valid reference queries is recorded.",
                "The nominal target uses Small padding/stride and the official range-bin interval.",
                "Weighted sample centroids are not the location of the final attention output.",
                "Sampling offsets are measured, but learned compensation cannot be inferred from parameter means alone.",
                "Other sources of geometric error, feature mixing, and GT alignment are not tested.",
            ],
        }
        for idx, name in enumerate(self.modules):
            sample = self.frames.get(name)
            if not sample or "offsets" not in sample or "logits" not in sample:
                raise RuntimeError(f"No complete cross-attention hook capture for {name}")
            ref = sample["reference"].numpy()
            points = self._reshape_samples(name, sample)
            metrics, arrays = analyze_samples(
                ref, points[0], points[1], sample["spatial_shapes"].numpy()
            )
            arrays["query_indices"] = sample["query_indices"].cpu().numpy()
            arrays["spatial_shapes"] = sample["spatial_shapes"].numpy()
            path = out_dir / f"cross_attention_layer_{idx}.npz"
            np.savez_compressed(path, **arrays)
            metrics["npz_file"] = str(path)
            metrics["reference_batch_in_forward"] = sample["reference_batch"]
            metrics["offset_linear_batch_in_forward"] = points[2]
            if metrics["reference_batch_in_forward"] != points[2]:
                metrics["reference_batch_warning"] = (
                    "Reference tensor batch differs from query batch. Only batch 0 was "
                    "analyzed; inspect VoxFormer broadcasting before interpreting output."
                )
            manifest["modules"][name] = metrics
        output = out_dir / "summary.json"
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"Saved: {output.resolve()}")

    @staticmethod
    def _reshape_samples(name, sample):
        mod = sample.get("module")
        if mod is None:
            raise RuntimeError(f"{name}: missing attention module metadata")
        offsets = sample["offsets"]
        logits = sample["logits"]
        n = offsets.shape[0]
        h, l, p = mod.num_heads, mod.num_levels, mod.num_points
        expected = h * l * p
        if offsets.shape[1] != expected * 3 or logits.shape[1] != expected:
            raise RuntimeError(f"{name}: unexpected offset/logit dimensions")
        return (
            offsets.reshape(n, h, l, p, 3).numpy(),
            logits.reshape(n, h, l, p).numpy(),
            sample["offset_linear_batch"],
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint", help="Locally trained/trusted checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--index", type=int, default=0, help="0-based dataset sample index")
    parser.add_argument("--queries", type=int, default=256,
                        help="Valid queries to record, spread across the grid")
    parser.add_argument("--out-dir", default="work_dirs/cross_attention_sampling")
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be >= 0")

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.occ_plugin.datasets.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("This inference diagnostic requires a CUDA GPU")
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", args.config)
        module_path = ".".join(Path(plugin_dir).parent.parts)
        importlib.import_module(module_path)
    test_data_cfg = copy.deepcopy(cfg.data[args.split])
    test_data_cfg.test_mode = True
    if args.index >= len(build_dataset(copy.deepcopy(test_data_cfg))):
        parser.error(f"--index is past the end of {args.split} dataset")
    dataset = build_dataset(test_data_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().cuda()
    capture = CrossAttentionCapture(args.queries)
    capture.attach(model)
    for name, module in model.named_modules():
        if name in capture.modules:
            capture.frames.setdefault(name, {})["module"] = module
    wrapped = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    try:
        data = None
        for i, batch in enumerate(loader):
            if i == args.index:
                data = batch
                break
        if data is None:
            raise RuntimeError("Data loader ended before the requested index")
        print(f"Running {args.split} dataset index {args.index}, modules: {capture.modules}")
        with torch.inference_mode():
            wrapped(return_loss=False, rescale=True, **data)
        # The pre-hook replaces per-call data; attach metadata for shape inference.
        capture.save(args.out_dir, args.config, args.checkpoint, args.split, args.index)
    finally:
        capture.close()


if __name__ == "__main__":
    main()
