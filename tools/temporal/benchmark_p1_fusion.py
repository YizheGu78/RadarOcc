#!/usr/bin/env python3
"""Synthetic P1 forward/backward cost, not full-model or accuracy evaluation."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from temporal import build_temporal_plugin
from temporal.core.ego_alignment import RadarBinGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins-path', required=True)
    parser.add_argument('--mode', choices=('ra_adta', 'ego_attention'), default='ra_adta')
    parser.add_argument('--points', type=int, default=43750)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.points, args.steps) <= 0:
        parser.error('--points and --steps must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; explicitly pass --device cpu for a CPU smoke check')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bins = RadarBinGeometry.from_file(args.bins_path)
    # Match the baseline's 175 range groups / 250 points for default N. These
    # are synthetic tokens, not real Top-K measurements or an IoU experiment.
    ranges = min(175, len(bins.bins[0]), args.points)
    coords = np.zeros((1, 4, args.points, 3), dtype=np.int64)
    coords[..., 0] = np.arange(args.points) * ranges // args.points
    for axis in (1, 2):
        coords[..., axis] = rng.integers(len(bins.bins[axis]), size=(1, 4, args.points))
    coords = torch.as_tensor(coords, device=device)
    features = torch.rand(1, 4, args.points, 8, device=device) + 1
    poses = torch.eye(4, device=device).repeat(1, 4, 1, 1)
    poses[0, :, 0, 3] = torch.arange(4, device=device) * 0.25
    offsets = torch.tensor([[-3, -2, -1, 0]], device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    attention = {} if args.chunk_size is None else dict(chunk_size=args.chunk_size)
    plugin = build_temporal_plugin(dict(mode=args.mode,
        alignment=dict(bins_path=args.bins_path), attention=attention)).to(device).train()
    optimizer = torch.optim.AdamW(plugin.parameters(), lr=3e-4)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def run_step():
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        start = time.perf_counter()
        out, debug = plugin(features, coords, poses, offsets, valid_mask=mask)
        synchronize()
        forward_end = time.perf_counter()
        if not torch.isfinite(out).all():
            raise RuntimeError('Nonfinite P1 output')
        if not debug['used_history']:
            raise RuntimeError('Synthetic input has no usable history; increase --points')
        (out / features[:, -1]).square().mean().backward()
        synchronize()
        backward_end = time.perf_counter()
        gradients = {name: float(parameter.grad.detach().norm())
                     for name, parameter in plugin.named_parameters() if parameter.grad is not None}
        optimizer.step()
        return dict(forward_seconds=forward_end - start,
                    backward_seconds=backward_end - forward_end,
                    queries_with_history=debug['queries_with_history'], gradient_norms=gradients)

    warmup = run_step()  # Zero residual learns first; upstream grads appear next step.
    print('Warmup complete: forward {:.3f}s, backward {:.3f}s'.format(
        warmup['forward_seconds'], warmup['backward_seconds']), file=sys.stderr, flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    measurements = [run_step() for _ in range(args.steps)]
    print(json.dumps(dict(mode=args.mode, synthetic=True, points=args.points,
        device=str(device), torch_version=torch.__version__,
        warmup=warmup, measurements=measurements,
        peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                            if device.type == 'cuda' else None)), indent=2))


if __name__ == '__main__':
    main()
