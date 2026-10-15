"""Physical, query-centred radar coordinates for RA-ADTA (angles in radians)."""
import itertools

import numpy as np


def _spacing_table(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError('RA-ADTA requires at least two physical bins on every axis')
    order = np.argsort(values)
    values = values[order]
    gaps = np.diff(values)
    if not np.isfinite(gaps).all() or (gaps <= 0).any():
        raise ValueError('Physical bin tables must contain distinct finite values')
    # Central local interval; one-sided interval at either endpoint.
    spacing = np.r_[gaps[0], (gaps[:-1] + gaps[1:]) / 2, gaps[-1]]
    return values, spacing


class RangeAnisotropicGeometry:
    def __init__(self, radar_geometry, alpha_range=3.0, alpha_azimuth=3.0,
                 alpha_elevation=3.0, min_angular_scale_m=0.25):
        self.alpha = np.array([alpha_range, alpha_azimuth, alpha_elevation],
                              dtype=np.float64)
        self.minimum = float(min_angular_scale_m)
        if (not np.isfinite(self.alpha).all() or (self.alpha <= 0).any()
                or not np.isfinite(self.minimum) or self.minimum <= 0):
            raise ValueError('Anisotropic scales must be finite and positive')
        bins = list(radar_geometry.bins)
        bins[1] = bins[1] * radar_geometry.azimuth_sign
        bins[2] = bins[2] * radar_geometry.elevation_sign
        self.tables = [_spacing_table(axis) for axis in bins]

    def query_geometry(self, xyz):
        """Return local basis [...,3,3] (columns er/eaz/eel) and scales [...,3].

        Bin intervals come from the loaded physical table, including nonuniform
        spacing; no Small feature-sampling approximation is used.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Query coordinates must be finite [...,3]')
        x, y, z = np.moveaxis(xyz, -1, 0)
        horizontal = np.hypot(x, y)
        r = np.hypot(horizontal, z)
        az, el = np.arctan2(y, x), np.arctan2(z, horizontal)
        ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        er = np.stack((ce * ca, ce * sa, se), axis=-1)
        ea = np.stack((-sa, ca, np.zeros_like(sa)), axis=-1)
        ee = np.stack((-se * ca, -se * sa, ce), axis=-1)
        basis = np.stack((er, ea, ee), axis=-1)
        dr, da, de = [np.interp(v, table, step) for v, (table, step) in
                      zip((r, az, el), self.tables)]
        scales = np.stack((self.alpha[0] * dr,
                           np.maximum(self.alpha[1] * horizontal * da, self.minimum),
                           np.maximum(self.alpha[2] * r * de, self.minimum)), axis=-1)
        return basis, scales


def normalized_displacement(points, query, basis, scales):
    """World displacement -> query's dimensionless local radar coordinates."""
    return np.einsum('...i,...ij->...j', points - query, basis) / scales


def sampling_anchors(count, offset_bound):
    """Deterministic cube coverage: centre, corners, then edges/faces.

    These anchors only diversify the discrete candidate pool. Learned offsets
    remain continuous and are not snapped to this lattice.
    """
    side = max(3, int(np.ceil(count ** (1 / 3))))
    grid = np.asarray(list(itertools.product(np.linspace(-1, 1, side), repeat=3)))
    selected = [int(np.argmin((grid ** 2).sum(-1)))]
    distance = ((grid - grid[selected[0]]) ** 2).sum(-1)
    for _ in range(1, count):
        ix = int(np.argmax(distance))
        selected.append(ix)
        distance = np.minimum(distance, ((grid - grid[ix]) ** 2).sum(-1))
        distance[selected] = -1
    return grid[selected] * offset_bound
