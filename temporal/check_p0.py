"""Read-only real-frame P0 forward/backward check; does not load occupancy GT.

Run from repository root after sourcing use_radarocc5060.sh:
    python -m temporal.check_p0 --method centerformer --device cpu
"""
import argparse
import importlib
import json

import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['centerformer', 'rodnet'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--index', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile('temporal/'+args.method+'/config_p0.py')
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    from projects.occ_plugin.occupancy.detectors.radarocc_self_small import RadarOcc_small
    from temporal.adapters import P1TopKAdapter
    from temporal.centerformer.fusion import CenterFormerP0
    from temporal.rodnet.fusion import RODNetP0
    dataset_cfg = dict(cfg.data.val)
    dataset_cfg.update(pipeline=[], test_mode=True)
    dataset = build_dataset(dataset_cfg)
    info = dataset.get_temporal_info(args.index)
    prepared = []
    for path in info['radar_paths']:
        raw = dataset._load_sparse_radar_npz(path)
        raw = {k: torch.as_tensor(v, device=args.device).unsqueeze(0) for k, v in raw.items()}
        prepared.append(RadarOcc_small._prepare_p1_frame(None, raw))
    inputs = P1TopKAdapter()([p[0] for p in prepared], [p[1] for p in prepared])
    kwargs = dict(poses=torch.as_tensor(np.asarray(info['ego_poses'])[None]),
                  frame_offsets=torch.as_tensor(info['frame_offsets'][None]),
                  valid_mask=torch.as_tensor(info['valid_mask'][None]))
    options = dict(cfg.model.p0_temporal_cfg); options.pop('method')
    cls = CenterFormerP0 if args.method == 'centerformer' else RODNetP0
    module = cls(**options).to(args.device).train()
    features = inputs['features']
    original = features.clone()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        out, debug = module(features, inputs['coords'], **kwargs)
        if step == 0:
            assert torch.equal(out, features[:, -1]), 'Initial identity failed'
        assert torch.isfinite(out).all()
        assert torch.equal(out[..., 3:6], features[:, -1, :, 3:6])
        # A scaled descriptor objective tests the P0 graph, not occupancy quality.
        loss = torch.log1p(out[..., [0, 1, 2, 6, 7]].abs()).mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
        losses.append(float(loss.detach()))
        optimizer.step()
    layer = module.attention.sampling_offsets if args.method == 'centerformer' else module.tdc.conv_offset
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0, 'No offset gradient'
    assert torch.equal(features, original), 'Source descriptors were mutated'
    print(json.dumps(dict(method=args.method, points=features.shape[2], identity=True,
                          finite_gradients=True, offset_gradient=float(layer.weight.grad.abs().sum()),
                          descriptor_objectives=losses, diagnostics=debug), indent=2))


if __name__ == '__main__':
    main()
