# Source and license notices

The Python CPU port in `autoware/costmap.py` is derived from the files recorded
in `SOURCE_MANIFEST.json`. The original files are retained verbatim in `vendor/`,
including their copyright, license conditions and disclaimers.

- Copyright 2021, 2024 Tier IV, Inc.; Apache License 2.0.
- Copyright (c) 2008, 2013, Willow Garage, Inc. All rights reserved;
  BSD 3-Clause license, reproduced in the vendored source headers.
- Original authors listed in upstream headers: Eitan Marder-Eppstein,
  David V. Lu!!
- Translation and standalone dataset integration added for RadarOcc in 2026.

The Apache license text is in `vendor/LICENSE-Autoware`. The BSD terms and
disclaimer in `vendor/nav2_costmap_2d.hpp` also apply to the translated
`raytraceLine` / `bresenham2D` implementation.

Autoware source:
https://github.com/YizheGu78/autoware_universe/tree/1c93b65555410c8c7b8299dab2f48320d1cb19e4/perception/autoware_probabilistic_occupancy_grid_map

Autoware inherits Bresenham from its external Nav2 dependency. This snapshot
uses the `humble` implementation of `nav2_costmap_2d/costmap_2d.hpp`; the exact
file hash is recorded. The fork does not pin a Nav2 source commit, so this is
an explicit dependency selection, not a claim about the installed ROS binary.

GM2019 reference:
Liat Sless, Gilad Cohen, Bat El Shlomo, Shaul Oron,
*Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters
using Semantic Segmentation*, ICCV Workshops 2019, arXiv:1904.00415v2.
The earlier v1 title is *Self-Supervised Occupancy Grid Learning from Sparse Radar
for Autonomous Driving*.
https://arxiv.org/abs/1904.00415

Only the classical Bayesian/ISM baseline in section 3.1, equations (3)-(4), is
implemented here. No GM official source code was used or is claimed. DBSCAN,
the 42-D descriptor schema, random forest, and the height-layer adapter are
RadarOcc project additions, not methods claimed by the GM2019 authors.

`adapters/paths.py`, `adapters/pose_reader.py`, and
`evaluation/radarocc_metrics.py` snapshot the IO/metric conventions of RadarOcc
commit `990ddee58285708f9692ee6e2214d5c69dde244f`. They do not import, change,
or execute the older `tradition` pipeline.
