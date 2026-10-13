"""Audit Small's base reference coordinates; no model/checkpoint execution.

Compares continuous encoder receptive-field centres with CUDA sampling locations.
This is a nominal geometry audit, not the location of learned feature evidence:
convolutions mix neighbours, self-attention mixes features, and cross-attention
adds learned offsets. The official table defines coordinates, not object truth.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = 'projects/occ_plugin/occupancy/detectors/radarocc_self_small.py'
ENCODER = 'projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py'
KERNEL = 'third_party/VoxFormer/deform_attn_3d/csrc/ms_deform_attn_cuda_kernel.cuh'


# The historical implementation is kept as comments in RadarOcc Small.
# Only examine ACTIVE AST assignments, so commented lines cannot fool the audit.
FORMULAE = {
    'legacy': {
        'r_res': '0.46 * scale',
        'rs_index': '(rs / r_res) / range_shape',
        'thetas_index': '(((thetas) / theta_res) + (azi_shape//2)) / azi_shape',
        'phis_index': '(((phis) / phi_res) + ele_shape//2) / ele_shape',
    },
    'corrected': {
        'r_res': '0.462890625 * scale',
        'rs_index': '((rs / r_res) + 0.5) / range_shape',
        'thetas_index': '((thetas / theta_res) + (azi_shape // 2) + 0.25) / azi_shape',
        'phis_index': '((phis / phi_res) + (ele_shape // 2) + 0.25) / ele_shape',
    },
}


def detect_detector_formula(tree):
    """Identify the active RadarOcc Small reference formula from its AST."""
    methods = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == 'get_reference_points_spherical']
    if len(methods) != 1:
        raise ValueError('Expected exactly one get_reference_points_spherical method')
    assignments = {}
    for node in methods[0].body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    matched = []
    for formula, expected in FORMULAE.items():
        if all(
            name in assignments and
            ast.dump(assignments[name], include_attributes=False) ==
            ast.dump(ast.parse(expression, mode='eval').body,
                     include_attributes=False)
            for name, expression in expected.items()
        ):
            matched.append(formula)
    if len(matched) != 1:
        found = {name: ast.unparse(assignments[name])
                 for name in FORMULAE['corrected'] if name in assignments}
        raise ValueError(
            'Unsupported or mixed active Small reference formula; '
            f'cannot audit safely. Found: {found}')
    return matched[0]


def verify_sources(root):
    """Fail on unknown model geometry instead of auditing obsolete constants."""
    checks = {
        DETECTOR: ['theta_res=1*scale*(torch.pi/180)',
                   'phi_res=1*scale*(torch.pi/180)',
                   'spherical_shape=[h,w,z],scale=2'],
        ENCODER: ['ele_ind=coors[:,1]*2+(56-37*2)//2',
                  'range_ind=coors[:,2]*2', 'azi_ind=coors[:,3]*2+(512-107*2)//2'],
    }
    active_formula = None
    hashes = {}
    for path, needles in checks.items():
        content = (root / path).read_text()
        tree = ast.parse(content)
        # Only active AST nodes; comments cannot satisfy these checks.
        if path == ENCODER:
            tree = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'RadarEncV8small')
        text = ''.join(ast.unparse(tree).split())
        if path == DETECTOR:
            active_formula = detect_detector_formula(tree)
            # Check h/w/z semantically rather than requiring list or tuple syntax.
            # Do not silently accept a changed feature shape: audit() assumes 128x128x14.
            spatial_assignments = []
            shape_matches = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, (ast.Tuple, ast.List)):
                        continue
                    names = [elt.id if isinstance(elt, ast.Name) else None
                             for elt in target.elts]
                    if names != ['h', 'w', 'z']:
                        continue
                    spatial_assignments.append(ast.unparse(node))
                    try:
                        dims = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(dims, (list, tuple)) and list(dims) == [128, 128, 14]:
                        shape_matches = True
            if not shape_matches:
                raise ValueError(
                    f'{path}: expected an active h,w,z=(128,128,14) assignment; '
                    f'found: {spatial_assignments or "none"}. '
                    'This audit hardcodes shape (128,128,14); check the actual model '
                    'rather than skipping this guard.'
                )
        for needle in needles:
            if needle not in text:
                raise ValueError(f'{path}: unsupported formula {needle}; re-audit source')
        if path == ENCODER and text.count('stride=2,padding=1') != 2:
            raise ValueError('Expected two centred stride-2 convolution blocks')
        hashes[path] = hashlib.sha256(content.encode()).hexdigest()
    kernel = root / KERNEL
    if kernel.exists():
        content = kernel.read_text()
        compact = ''.join(content.split())
        for axis in ('h', 'w', 'z'):
            if f'{axis}_im=loc_{axis}*spatial_{axis}-0.5' not in compact:
                raise ValueError('CUDA sampler centre convention changed')
        hashes[KERNEL] = hashlib.sha256(content.encode()).hexdigest()
    return hashes, kernel.exists(), active_formula


def load_bins(path):
    d = loadmat(path)
    r, a, e = [np.asarray(d[k], dtype=float).ravel()
               for k in ('arrRange', 'arrAzimuth', 'arrElevation')]
    if tuple(map(len, (r, a, e))) != (256, 107, 37):
        raise ValueError('Expected official 256/107/37 table')
    for values in (r, a, e):
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError('Tables must be finite and strictly increasing')
        if not np.allclose(np.diff(values), np.diff(values)[0]):
            raise ValueError('This analytic audit requires uniform tables')
    return r, a, e


def xyz(spherical):
    r, a, e = spherical.T
    a, e = np.deg2rad(a), np.deg2rad(e)
    return np.column_stack((r*np.cos(e)*np.cos(a),
                            r*np.cos(e)*np.sin(a), r*np.sin(e)))


def audit(indices, bins, formula='corrected'):
    """All vectors ordered range, azimuth, elevation; errors signed query-true.

    formula='corrected' matches the active reference coordinates after the fix.
    formula='legacy' independently reproduces the original calculations.
    """
    if formula not in FORMULAE:
        raise ValueError(f'Unsupported formula: {formula}')
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 3 or not np.isfinite(indices).all():
        raise ValueError('Expected finite [N,3] R/A/E indices')
    if not np.array_equal(indices, indices.astype(np.int64)):
        raise ValueError('Indices must be integers')
    indices = indices.astype(np.int64)
    for k, values in enumerate(bins):
        if (indices[:, k] < 0).any() or (indices[:, k] >= len(values)).any():
            raise ValueError('Indices exceed official table')
    true_spherical = np.column_stack([v[indices[:, k]] for k, v in enumerate(bins)])
    # Encoder padding and two k3/p1/s2 convolutions: centre j maps to 4*j.
    padded = indices * 2 + np.array([0, 149, -9])
    expected_feature = padded / 4.0
    shape = np.array([128, 128, 14])  # R/A/E; dimensions passed to attention
    if formula == 'corrected':
        scale_steps = np.array([0.462890625 * 2., 2., 2.])
        # Offset is expressed in feature-index units, before normalization.
        # CUDA subtracts 0.5 from each dimension's normalized coordinate.
        shifts = np.array([0.5, 64.25, 7.25])
    else:
        scale_steps = np.array([0.46 * 2., 2., 2.])
        shifts = np.array([0., 64., 7.])
    pre_normalized = true_spherical / scale_steps + shifts
    normalized = pre_normalized / shape
    # Pinned VoxFormer CUDA kernel: f = normalized * size - 0.5.
    query_feature = normalized * shape - 0.5
    raw_query = (4 * query_feature - np.array([0, 149, -9])) / 2
    origins = np.array([v[0] for v in bins])
    steps = np.array([v[1]-v[0] for v in bins])
    query_spherical = origins + raw_query * steps
    # Do not disguise invalid encoder coordinates or table extrapolation.
    encoder_valid = ((padded >= 0) & (padded < np.array([512, 512, 56]))).all(axis=1)
    query_inside_table = ((raw_query >= 0) & (raw_query <= np.array([255,106,36]))).all(axis=1)
    query_inside_feature = ((query_feature >= 0) & (query_feature <= shape-1)).all(axis=1)
    true_xyz, query_xyz = xyz(true_spherical), xyz(query_spherical)
    valid = encoder_valid & query_inside_table & query_inside_feature & (true_spherical[:,0] > 0)
    return dict(indices=indices, true_spherical=true_spherical,
                query_spherical=query_spherical, expected_feature=expected_feature,
                query_feature=query_feature, feature_error=query_feature-expected_feature,
                spherical_error=query_spherical-true_spherical,
                true_xyz=true_xyz, query_xyz=query_xyz,
                xyz_error=query_xyz-true_xyz,
                euclidean_error=np.linalg.norm(query_xyz-true_xyz, axis=1),
                encoder_valid=encoder_valid, query_inside_table=query_inside_table,
                query_inside_feature=query_inside_feature, valid=valid)


def stats(values):
    return dict(mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values,95)), max=float(np.max(values)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bins', required=True, type=Path, help='Official info_arr.mat (degrees)')
    p.add_argument('--radar-npz', type=Path, help='Optional real NPZ; first 175*250 candidates')
    p.add_argument('--source-root', type=Path, default=ROOT)
    p.add_argument('--out-dir', type=Path, default=Path('work_dirs/small_geometry_audit'))
    p.add_argument('--compare-legacy', action='store_true',
                   help='Also calculate old-formula errors on exactly the same valid samples')
    args = p.parse_args()
    hashes, local_kernel_verified, active_formula = verify_sources(args.source_root)
    print('Detected ACTIVE detector formula:', active_formula)
    bins = load_bins(args.bins)
    if args.radar_npz:
        with np.load(args.radar_npz, allow_pickle=False) as d:
            arrays = [d[k] for k in ('range_ind','azimuth_ind','elevation_ind')]
            if any(x.shape != (64000,) for x in arrays):
                raise ValueError('Expected the 64000-candidate training NPZ')
            indices = np.column_stack(arrays)[:175*250]
            if not np.array_equal(indices[:,0],np.repeat(np.arange(175),250)):
                raise ValueError('NPZ is not grouped by range as Small expects')
    else:
        indices = np.stack(np.meshgrid(np.arange(175),np.arange(107),np.arange(37),
                                       indexing='ij'),axis=-1).reshape(-1,3)
    result = audit(indices,bins,formula=active_formula)
    mask = result['valid']
    if not mask.any():
        raise ValueError('No valid points for statistics')
    report = dict(scope='Nominal base-reference geometry, zero learned sampling offsets; not prediction/GT error',
                  active_detector_formula=active_formula,
                  source_sha256=hashes, local_cuda_source_verified=local_kernel_verified,
                  sampler_source='YizheGu78/VoxFormer@2b25090051643aecf331914bb4bfbe3b5b59e8ff',
                  sampler_convention='f = normalized * size - 0.5',
                  table_sha256=hashlib.sha256(args.bins.read_bytes()).hexdigest(),
                  table_steps_RAE=[float(v[1]-v[0]) for v in bins],
                  input=str(args.radar_npz) if args.radar_npz else 'All angle bins, first 175 ranges; unweighted',
                  total=len(indices), valid=int(mask.sum()),
                  invalid_encoder_coordinates=int((~result['encoder_valid']).sum()),
                  outside_physical_table=int((~result['query_inside_table']).sum()),
                  outside_feature_centres=int((~result['query_inside_feature']).sum()),
                  xyz_error_m=stats(result['euclidean_error'][mask]),
                  signed_spherical_error_RAE={name:stats(result['spherical_error'][mask,k])
                                             for k,name in enumerate(('range_m','azimuth_deg','elevation_deg'))})
    examples = []
    for r in (10.,30.,50.,80.):
        ir = int(np.argmin(abs(bins[0]-r)))
        sample = audit([[ir,53,18]],bins,formula=active_formula)
        examples.append(dict(range_m=float(sample['true_spherical'][0,0]),
                             signed_range_error_m=float(sample['spherical_error'][0,0]),
                             azimuth_error_deg=float(sample['spherical_error'][0,1]),
                             elevation_error_deg=float(sample['spherical_error'][0,2]),
                             xyz_error_m=float(sample['euclidean_error'][0]),
                             within_first_175_ranges=bool(ir<175)))
    report['zero_angle_examples'] = examples
    if args.compare_legacy:
        legacy = audit(indices,bins,formula='legacy')
        common = legacy['valid'] & result['valid']
        if not common.any():
            raise ValueError('No samples valid under BOTH legacy and active formulas')
        report['comparison_same_valid'] = {
            'valid_for_both': int(common.sum()),
            'active_formula': active_formula,
            'active_xyz_error_m': stats(result['euclidean_error'][common]),
            'legacy_xyz_error_m': stats(legacy['euclidean_error'][common]),
            'active_signed_spherical_error_RAE': {
                name: stats(result['spherical_error'][common,k])
                for k,name in enumerate(('range_m','azimuth_deg','elevation_deg'))},
            'legacy_signed_spherical_error_RAE': {
                name: stats(legacy['spherical_error'][common,k])
                for k,name in enumerate(('range_m','azimuth_deg','elevation_deg'))},
        }
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.out_dir/'coordinates.npz',**result)
    with (args.out_dir/'examples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=examples[0].keys())
        writer.writeheader(); writer.writerows(examples)
    print(json.dumps(report,indent=2))
    print('Saved:',args.out_dir.resolve())
    if not local_kernel_verified:
        print('NOTE: local CUDA source absent; pinned repository convention used. Verify your installed extension matches.')


if __name__ == '__main__':
    main()