#!/usr/bin/env python3
"""Independent, CPU-only quality audit for P1 reflection-region proposals and links.

GT IDs are attached for offline scoring, never consumed by GT-free region
extraction or geometric match selection. Does not train/modify RadarOcc.
"""
import argparse
import csv
import itertools
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import ConvexHull

from tools.temporal_vis.benchmark_region_matching import (
    FrameData, gt_memberships, history_regions, label_regions, select_match)
from tools.temporal_vis.audit_dynamic_objects import transform
from temporal.core.ego_alignment import add_coordinate_argument

GATES = {
    "A_original": ((0.5, 1.0, 1.5), (4.0, 6.0, 8.0)),
    "B_y_only": ((4.0, 5.0, 3.0), (4.0, 6.0, 8.0)),
    "C_y_and_center": ((4.0, 5.0, 3.0), (4.0, 12.0, 16.0)),
}


def divide(n, d):
    return float(n / d) if d else None


def median(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return float(np.median(a)) if len(a) else None


def aabb_iou(low_a, high_a, low_b, high_b):
    """Exact IoU for TWO axis-aligned boxes only (3D proxy for oriented GT)."""
    a, b = np.asarray(low_a), np.asarray(high_a)
    c, d = np.asarray(low_b), np.asarray(high_b)
    inter = np.prod(np.maximum(0, np.minimum(b, d) - np.maximum(a, c)))
    v1 = np.prod(np.maximum(0, b - a))
    v2 = np.prod(np.maximum(0, d - c))
    return divide(inter, v1 + v2 - inter)


def polygon_area(p):
    p = np.asarray(p, float)
    if len(p) < 3:
        return 0.0
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2)


def clip_halfplane(polygon, axis, boundary, keep_greater):
    """Sutherland-Hodgman convex polygon clipping."""
    poly = np.asarray(polygon, float).reshape(-1, 2)
    if len(poly) == 0:
        return poly
    out = []
    def val(p):
        return (p[axis] - boundary) * (1 if keep_greater else -1)
    a = poly[-1]
    va = val(a)
    for b in poly:
        vb = val(b)
        if (va >= -1e-10) != (vb >= -1e-10):
            t = va / (va - vb)
            out.append(a + t * (b - a))
        if vb >= -1e-10:
            out.append(b)
        a, va = b, vb
    return np.asarray(out, float).reshape(-1, 2)


def xy_polygon_iou(box_low, box_high, gt_corners):
    """Exact 2D IoU: radar-axis-aligned search box vs projected yaw/tilt GT hull."""
    corners = np.asarray(gt_corners)[:, :2]
    hull = corners[ConvexHull(corners).vertices]
    clipped = hull.copy()
    for axis, bound, greater in ((0, box_low[0], True),
                                 (0, box_high[0], False),
                                 (1, box_low[1], True),
                                 (1, box_high[1], False)):
        clipped = clip_halfplane(clipped, axis, bound, greater)
    intersection = polygon_area(clipped)
    union = polygon_area(hull) + np.prod(np.asarray(box_high[:2])-box_low[:2]) - intersection
    return divide(intersection, union)


def lidar_box_corners_radar(box, radar_to_lidar):
    """K-Radar labels use bottom-center z (same convention as benchmark)."""
    x, y, z, dx, dy, dz, yaw = np.asarray(box, float)
    corners = np.asarray(list(itertools.product((-0.5, 0.5), (-0.5, 0.5), (0, 1))), float)
    corners *= [dx, dy, dz]
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return transform(corners @ rot.T + [x, y, z], np.linalg.inv(radar_to_lidar))


def point_target_ids(membership, tokens):
    """-1 (unlabeled), -2 (overlap), duplicate GT ID -> None, never clutter."""
    tokens = np.asarray(tokens).astype(str)
    counts = Counter(tokens.tolist())
    out = np.empty(len(membership), dtype=object)
    out[:] = None
    for index in np.unique(membership):
        if index >= 0:
            token = str(tokens[int(index)])
            if counts[token] == 1:
                out[membership == index] = token
    return out


def fraction_for_target(indices, tags, target):
    indices = np.asarray(indices, dtype=int)
    if target is None or not len(indices):
        return None, 0
    hits = int(np.count_nonzero(tags[indices] == target))
    return hits / len(indices), hits


def quality_for_region(region, target, xyz_tags, xy_tags, corners_by_id):
    support = np.asarray(region["support_point_ids"], dtype=np.int64)
    enclosed = np.asarray(region["box_point_ids"], dtype=np.int64)
    support_xyz, support_xyz_hits = fraction_for_target(support, xyz_tags, target)
    box_xyz, box_xyz_hits = fraction_for_target(enclosed, xyz_tags, target)
    support_xy, support_xy_hits = fraction_for_target(support, xy_tags, target)
    box_xy, box_xy_hits = fraction_for_target(enclosed, xy_tags, target)
    # A target ID may be absent; record total GT-assigned evidence nonetheless.
    known_xyz = sum(t is not None for t in xyz_tags[enclosed])
    known_xy = sum(t is not None for t in xy_tags[enclosed])
    iou_xy, iou_3d_proxy = None, None
    if target in corners_by_id:
        corners = corners_by_id[target]
        iou_xy = xy_polygon_iou(region["box_min"], region["box_max"], corners)
        iou_3d_proxy = aabb_iou(
            region["box_min"], region["box_max"], corners.min(0), corners.max(0))
    return dict(
        support_points=len(support), box_points=len(enclosed),
        support_purity_xyz=support_xyz, box_purity_xyz=box_xyz,
        support_purity_xy=support_xy, box_purity_xy=box_xy,
        support_target_xyz_points=support_xyz_hits, box_target_xyz_points=box_xyz_hits,
        support_target_xy_points=support_xy_hits, box_target_xy_points=box_xy_hits,
        box_known_xyz_points=int(known_xyz), box_known_xy_points=int(known_xy),
        box_unlabeled_or_ambiguous_xyz_points=int(len(enclosed)-known_xyz),
        box_xy_gt_iou=iou_xy, box_3d_aabb_proxy_iou=iou_3d_proxy,
        center_x=float(region["center"][0]), center_y=float(region["center"][1]),
        center_z=float(region["center"][2]), size_x=float(region["box_size"][0]),
        size_y=float(region["box_size"][1]), size_z=float(region["box_size"][2]),
    )


def make_frame_loader(data, args):
    @lru_cache(maxsize=20)
    def load(scene, order):
        info = data.info_index[(scene, order)]
        frame = data.frame(scene, order)  # same source/proposals/loose labels as original benchmark
        path = data.radar_path(info)
        with np.load(path, allow_pickle=False) as src:
            n = 175 * 250
            az = np.asarray(src["azimuth_ind"][:n], int)
            el = np.asarray(src["elevation_ind"][:n], int)
        coords = np.stack([np.repeat(np.arange(175), 250), az, el], -1)
        xyz = data.geometry.to_xyz(coords)
        bbox_index = int(info["occ_frame_idx"])
        bbox_dir = args.bbox_root / str(scene) / "bbox"
        boxes = np.load(bbox_dir / f"bbox{bbox_index}.npy", allow_pickle=False).reshape(-1, 7)
        tokens = np.load(bbox_dir / f"bbox_token{bbox_index}.npy", allow_pickle=False).reshape(-1).astype(str)
        if len(boxes) != len(tokens):
            raise ValueError(f"GT box/token mismatch {scene}/{order}")
        xyz_members = gt_memberships(xyz, boxes, data.extrinsic, "xyz")
        xy_members = gt_memberships(xyz, boxes, data.extrinsic, "xy")
        xyz_tags = point_target_ids(xyz_members, tokens)
        xy_tags = point_target_ids(xy_members, tokens)
        strict, _ = label_regions(
            frame["accepted"], xyz_members, tokens, purity=args.strict_purity,
            min_gt_points=args.strict_min_points, return_diagnostics=True)
        token_counts = Counter(tokens.tolist())
        corners = {str(t): lidar_box_corners_radar(box, data.extrinsic)
                   for t, box in zip(tokens, boxes) if token_counts[str(t)] == 1}
        quality = {}
        for region in frame["accepted"]:
            rid = int(region["id"])
            target = frame["labels"].get(rid)
            quality[rid] = quality_for_region(region, target, xyz_tags, xy_tags, corners)
        # Object-level coverage uses ONLY targets with actual XYZ radar returns in
        # the analysis RoI. Separately retain all annotated GT objects, even if
        # no radar returns are present; do not hide proposal failures.
        eligible = (np.isfinite(xyz).all(1) &
                    (xyz >= data.config.lower).all(1) &
                    (xyz < data.config.upper).all(1))
        union_support = np.zeros(len(xyz), dtype=bool)
        union_box = np.zeros(len(xyz), dtype=bool)
        for region in frame["accepted"]:
            union_support[np.asarray(region["support_point_ids"], int)] = True
            union_box[np.asarray(region["box_point_ids"], int)] = True
        objects = []
        for token, multiplicity in Counter(tokens.tolist()).items():
            if multiplicity != 1:
                continue
            hit = (xyz_tags == token) & eligible
            n = int(hit.sum())
            objects.append(dict(
                scene=scene, order=order, gt_id=token,
                gt_has_3d_p1_points=n > 0, n_xyz_p1_points=n,
                support_covered_xyz_points=int((hit & union_support).sum()),
                box_covered_xyz_points=int((hit & union_box).sum()),
                support_xyz_coverage=divide((hit & union_support).sum(), n),
                box_xyz_coverage=divide((hit & union_box).sum(), n),
                has_assigned_loose_region=token in frame["labels"].values(),
                has_assigned_strict_region=token in strict.values()))
        frame["objects"] = objects
        frame["strict_labels"] = strict
        frame["quality"] = quality
        return frame
    return load


def matching_rows(data, load, scene, selected_positions):
    rows = []
    for pos in selected_positions:
        infos = data.scenes[scene]
        order = int(infos[pos]["order_in_scene"])
        current = load(scene, order)
        for slot in (1, 2, 3):
            if pos < slot:
                continue
            history_order = int(infos[pos-slot]["order_in_scene"])
            previous = load(scene, history_order)
            aligned = history_regions(current, previous, data.extrinsic)
            for region in current["accepted"]:
                rid = int(region["id"])
                current_id = current["labels"].get(rid)
                current_strict_id = current["strict_labels"].get(rid)
                loose_true = (current_id is not None and any(
                    previous["labels"].get(h["id"]) == current_id for h in aligned))
                strict_true = (current_strict_id is not None and any(
                    previous["strict_labels"].get(h["id"]) == current_strict_id for h in aligned))
                for name, (ys, centers) in GATES.items():
                    matched = select_match(region, aligned, slot, 1.0, 0.0,
                                           ys, centers, 0.5, "xy")
                    hid = int(matched["id"]) if matched else None
                    historical_id = previous["labels"].get(hid) if hid is not None else None
                    historical_strict_id = previous["strict_labels"].get(hid) if hid is not None else None
                    same_loose = bool(hid is not None and current_id is not None
                                      and current_id == historical_id)
                    same_strict = bool(hid is not None and current_strict_id is not None
                                       and current_strict_id == historical_strict_id)
                    curq = current["quality"][rid]
                    histq = previous["quality"].get(hid) if hid is not None else None
                    pair_iou = (aabb_iou(region["box_min"][:2], region["box_max"][:2],
                                matched["aligned_center"][:2] - matched["aligned_size"][:2]/2,
                                matched["aligned_center"][:2] + matched["aligned_size"][:2]/2)
                                if matched else None)
                    rows.append(dict(
                        scene=scene, order=order, history_order=history_order, slot=slot,
                        configuration=name, current_region=rid, history_region=hid,
                        current_id=current_id, history_id=historical_id,
                        current_strict_id=current_strict_id, history_strict_id=historical_strict_id,
                        scorable=current_id is not None, has_true_history=bool(loose_true),
                        has_true_history_strict=bool(strict_true), predicted=hid is not None,
                        same_id_loose=same_loose, same_id_strict=same_strict,
                        unknown_history=bool(hid is not None and historical_id is None),
                        current_box_purity_xyz=curq["box_purity_xyz"],
                        history_box_purity_xyz=histq["box_purity_xyz"] if histq else None,
                        current_box_target_xyz_points=curq["box_target_xyz_points"],
                        history_box_target_xyz_points=histq["box_target_xyz_points"] if histq else None,
                        current_support_purity_xyz=curq["support_purity_xyz"],
                        history_support_purity_xyz=histq["support_purity_xyz"] if histq else None,
                        current_gt_xy_iou=curq["box_xy_gt_iou"],
                        history_gt_xy_iou=histq["box_xy_gt_iou"] if histq else None,
                        matched_pair_xy_aabb_iou=pair_iou,
                        matched_center_distance_xy=(
                            float(np.linalg.norm(np.asarray(region["center"][:2]) -
                                                 matched["aligned_center"][:2])) if matched else None),
                        confirmed_3d_target_points_both=bool(
                            same_loose and curq["box_target_xyz_points"] >= 3
                            and histq["box_target_xyz_points"] >= 3),
                    ))
    return rows


def summarize(rows):
    """Summarize only evaluable queries for old Recall, but expose ALL queries."""
    allq = len(rows)
    known = [r for r in rows if r["scorable"]]
    truth = [r for r in known if r["has_true_history"]]
    predicted = [r for r in known if r["predicted"]]
    correct = [r for r in predicted if r["same_id_loose"]]
    strict_truth = [r for r in rows if r["has_true_history_strict"]]
    strict_correct = [r for r in rows if r["same_id_strict"]]
    # Historic GT unknown predictions: not automatically incorrect or clutter.
    unknown = [r for r in predicted if r["unknown_history"]]
    wrong = [r for r in predicted if not r["same_id_loose"] and not r["unknown_history"]]
    p_values = [min(r["current_box_purity_xyz"], r["history_box_purity_xyz"])
                for r in correct if r["current_box_purity_xyz"] is not None
                and r["history_box_purity_xyz"] is not None]
    return dict(
        queries_all=allq, queries_labeled=len(known), queries_unlabeled=allq-len(known),
        eligible_loose=len(truth), predicted_labeled_queries=len(predicted),
        correct_loose=len(correct), wrong_known=len(wrong), unknown_history=len(unknown),
        recall_loose=divide(len(correct), len(truth)),
        precision_known=divide(len(correct), len(correct)+len(wrong)),
        precision_conservative=divide(len(correct), len(predicted)),
        unknown_history_rate=divide(len(unknown), len(predicted)),
        eligible_strict=len(strict_truth), correct_strict=len(strict_correct),
        recall_strict=divide(len(strict_correct), len(strict_truth)),
        loose_correct_strict_confirm_rate=divide(
            sum(r["same_id_strict"] for r in correct), len(correct)),
        loose_correct_3d_point_confirmation_rate=divide(
            sum(r["confirmed_3d_target_points_both"] for r in correct), len(correct)),
        matched_min_box_xyz_purity_median=median(p_values),
        matched_pair_xy_aabb_iou_median=median(
            r["matched_pair_xy_aabb_iou"] for r in correct),
        correct_current_gt_xy_iou_median=median(r["current_gt_xy_iou"] for r in correct),
        correct_history_gt_xy_iou_median=median(r["history_gt_xy_iou"] for r in correct),
    )


def decision(stats, object_summary=None):
    """Predeclared *screening*, not a calibrated proof of fusion performance."""
    n = stats["correct_loose"]
    if n < 30:
        return "INCONCLUSIVE: fewer than 30 same-ID links; enlarge sample"
    if object_summary is not None:
        n_objects = object_summary["gt_objects_with_xyz_radar"]
        assigned = object_summary["object_frame_assigned_loose_rate"]
        if n_objects < 30 or assigned is None:
            return "INCONCLUSIVE: insufficient annotated GT objects with 3D radar"
        if assigned < .25:
            return "PAUSE: <25% GT objects with radar evidence get any loose region"
    strict = stats["loose_correct_strict_confirm_rate"]
    evidence = stats["loose_correct_3d_point_confirmation_rate"]
    purity = stats["matched_min_box_xyz_purity_median"]
    if strict is None or evidence is None or purity is None:
        return "INCONCLUSIVE: missing scorable 3D evidence"
    if strict < 0.25 and evidence < 0.35 and purity < 0.15:
        return "PAUSE: audit calibration + examples; do not start expensive fusion training"
    object_ok = (object_summary is None or
                 object_summary["object_frame_assigned_loose_rate"] >= .5)
    if strict >= 0.60 and evidence >= 0.70 and purity >= 0.35 and object_ok:
        return "PILOT: reasonable to run small controlled fusion ablation"
    return "REVISE/INSPECT: no sufficient evidence to commit to fusion training"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_regions, all_matches, all_objects = [], [], []
    included = set()
    for ann_file in args.ann_files:
        local = argparse.Namespace(**vars(args))
        local.ann_file = ann_file
        local.seed_percentile = 97.
        local.seed_mad = 3.
        local.merge_space = "xy"
        local.xy_box_gap_m = (0.5, 0.5)
        local.box_size_m = (6., 3., 3.)
        local.max_box_size_m = (20., 6., 6.)
        local.no_xy_size_limit = False
        local.gt_space = "xy"
        local.match_space = "xy"
        local.gt_purity = 0.
        local.min_gt_points = 1
        local.y_gates = GATES["A_original"][0]
        local.center_gates = GATES["A_original"][1]
        data = FrameData(local)
        data.index()
        load = make_frame_loader(data, args)
        for scene, frames in data.scenes.items():
            if args.scenes and str(scene) not in args.scenes:
                continue
            if scene in included:
                raise ValueError(f"scene {scene} appears in multiple input annotations")
            included.add(scene)
            selected = [i for i, item in enumerate(frames)
                        if args.start_order <= int(item["order_in_scene"]) <= args.end_order]
            selected = selected[::args.stride][:args.max_frames_per_scene or None]
            needed = sorted(set(i-slot for i in selected for slot in (0, 1, 2, 3) if i >= slot))
            for i in needed:
                order = int(frames[i]["order_in_scene"])
                frame = load(scene, order)
                all_objects.extend(frame["objects"])
                for r in frame["accepted"]:
                    rid = int(r["id"])
                    q = frame["quality"][rid]
                    all_regions.append(dict(scene=scene, order=order, region_id=rid,
                        loose_xy_id=frame["labels"].get(rid),
                        strict_xyz_id=frame["strict_labels"].get(rid), **q))
            all_matches.extend(matching_rows(data, load, scene, selected))
            print(f"scene={scene} sampled={len(selected)} regions={len(all_regions)} "
                  f"links={len(all_matches)}", flush=True)
    if not included:
        raise ValueError("No scenes selected; check --ann-files / --scenes")
    write_csv(args.output_dir/"regions.csv", all_regions)
    write_csv(args.output_dir/"objects.csv", all_objects)
    write_csv(args.output_dir/"matches.csv", all_matches)
    radar_objects = [r for r in all_objects if r["gt_has_3d_p1_points"]]
    obj_summary = dict(
        gt_object_frame_instances=len(all_objects),
        gt_objects_with_xyz_radar=len(radar_objects),
        object_frame_assigned_loose_rate=divide(
            sum(r["has_assigned_loose_region"] for r in radar_objects), len(radar_objects)),
        object_frame_assigned_strict_rate=divide(
            sum(r["has_assigned_strict_region"] for r in radar_objects), len(radar_objects)),
        object_frame_any_box_coverage_rate=divide(
            sum(r["box_covered_xyz_points"] > 0 for r in radar_objects), len(radar_objects)),
        object_frame_any_support_coverage_rate=divide(
            sum(r["support_covered_xyz_points"] > 0 for r in radar_objects), len(radar_objects)),
        median_object_xyz_radar_coverage_by_boxes=median(
            r["box_xyz_coverage"] for r in radar_objects),
    )
    table = []
    for name in GATES:
        rows = [r for r in all_matches if r["configuration"] == name]
        stats = summarize(rows)
        table.append(dict(configuration=name, **stats, decision=decision(stats, obj_summary)))
    # Reconcile against the six-scene A/B/C values in the reference report.
    # A mismatch means the new audit results cannot be compared to the screenshot.
    regression = None
    expected_scenes = {"1", "2", "15", "22", "23", "55"}
    if (getattr(args, 'coordinate_convention', 'legacy_positive_v1') == 'legacy_positive_v1'
            and included == expected_scenes and args.start_order == 80
            and args.end_order == 200 and args.stride == 5
            and args.max_frames_per_scene == 0):
        reference = {"A_original": (234, 465), "B_y_only": (351, 465),
                     "C_y_and_center": (360, 465)}
        regression = {x["configuration"]: dict(
            expected_correct=reference[x["configuration"]][0],
            expected_eligible=reference[x["configuration"]][1],
            observed_correct=x["correct_loose"], observed_eligible=x["eligible_loose"],
            matches=(x["correct_loose"], x["eligible_loose"]) ==
                    reference[x["configuration"]]) for x in table}
        if not all(v["matches"] for v in regression.values()):
            for x in table:
                x["decision"] = "INVALID BASELINE: count mismatch; inspect annotation/GT/config"
    write_csv(args.output_dir/"comparison.csv", table)
    # Inspect lowest-purity links first. NOT a representative/random sample.
    bad = [r for r in all_matches if r["same_id_loose"]]
    bad.sort(key=lambda r: (
        min(r["current_box_purity_xyz"] or 0, r["history_box_purity_xyz"] or 0),
        r["scene"], r["order"]))
    write_csv(args.output_dir/"inspect_low_purity_links.csv", bad[:100])
    diagnostics = dict(
        source_ann_files=[str(p) for p in args.ann_files], scenes=sorted(included),
        sample=dict(start_order=args.start_order, end_order=args.end_order,
                    stride=args.stride, max_frames_per_scene=args.max_frames_per_scene),
        proposal_config=dict(seed_percentile=97, seed_mad=3, merge_space="xy",
            xy_box_gap_m=[.5,.5], box_size_m=[6,3,3], max_box_size_m=[20,6,6]),
        scoring=dict(original_gt_space="xy", original_gt_purity=0, original_min_gt_points=1,
            match_space="xy", strict_gt_space="xyz", strict_gt_purity=args.strict_purity,
            strict_min_gt_points=args.strict_min_points),
        counts=dict(regions=len(all_regions), links=len(all_matches), objects=len(all_objects)),
        object_coverage=obj_summary,
        per_configuration=table,
        baseline_regression=regression,
        limitations=[
            "This is post-hoc GT auditing; proposal extraction/matching never use GT.",
            "Box purity is the fraction of P1 radar returns in box inside THIS instance's 3D GT box; other returns are not necessarily clutter.",
            "Only regions labeled by permissive XY GT have an assigned target purity. Unlabeled regions are counted separately.",
            "AABB 3D IoU is a conservative geometry proxy, NOT oriented 3D IoU.",
            "XY GT IoU uses the true convex-hull projection; inter-frame region XY IoU uses aligned AABBs.",
            "Low 3D overlap can reflect sparse visible surfaces, motion, label/calibration error; check examples before rejection.",
            "Screening thresholds are heuristics, NOT validated universal significance tests.",
            "A promising audit requires a separate controlled RadarOcc IoU fusion ablation before a GO claim.",
        ])
    (args.output_dir/"report.json").write_text(json.dumps(
        diagnostics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    lines = ["# P1 反射区域可靠性审计", "",
        "独立检查 A/B/C；GT 只用于事后评价，不参与区域提取、合并或匹配。", "",
        "| 方案 | 查询数 | 宽松 Recall | 严格 XYZ Recall | 宽松正确对严格确认 | 3D 点证据确认 | 最差一侧 Box Purity 中位数 | 保守 P | 判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def pct(v):
        return "N/A" if v is None else f"{v:.1%}"
    for x in table:
        lines.append("| {configuration} | {queries_all} | {recall_loose} | {recall_strict} | "
                     "{loose_correct_strict_confirm_rate} | "
                     "{loose_correct_3d_point_confirmation_rate} | "
                     "{matched_min_box_xyz_purity_median} | "
                     "{precision_conservative} | {decision} |".format(
            **{k: pct(v) if k in (
                "recall_loose", "recall_strict", "loose_correct_strict_confirm_rate",
                "loose_correct_3d_point_confirmation_rate",
                "matched_min_box_xyz_purity_median", "precision_conservative") else v
               for k,v in x.items()}))
    if regression is not None:
        lines += ["", "历史六场景计数回归：" + ("PASS" if all(x["matches"] for x in regression.values()) else "FAIL，禁止根据审计结果推进")]
    lines += ["", "## GT 目标覆盖（与 A/B/C 匹配门限无关）", "",
              f"GT 目标-帧数：{len(all_objects)}；在分析 RoI 内有 3D 雷达点的目标-帧数：{len(radar_objects)}。",
              f"有 XYZ 雷达点的目标中，至少被一个接受框覆盖到 1 个目标点的比例：{pct(obj_summary['object_frame_any_box_coverage_rate'])}。",
              f"有 XYZ 雷达点的目标中，至少被一个接受区域支持点覆盖到 1 个目标点的比例：{pct(obj_summary['object_frame_any_support_coverage_rate'])}。",
              f"有 XYZ 雷达点的目标中，得到宽松 GT ID 区域的比例：{pct(obj_summary['object_frame_assigned_loose_rate'])}。",
              f"有 XYZ 雷达点的目标中，得到严格 GT ID 区域的比例：{pct(obj_summary['object_frame_assigned_strict_rate'])}。",
              "", "注意：3D 点证据确认 = 同 ID 的当前和历史框内分别至少有 3 个该目标的 3D GT 归属点。",
              "严格 XYZ ID = 支持点至少 3 个、Support Purity 至少 {:.0%}；其分母不同于宽松 Recall。".format(args.strict_purity),
              "最差一侧 Box Purity = min(当前框纯度, 历史框纯度)。未知/未标注反射不能直接称为杂波。",
              "低纯度样例见 inspect_low_purity_links.csv；需检查雷达与 GT 标定、物体可见性。",
              "自动判断只是筛查；真正继续/放弃须结合样例复核和下游 IoU ablation。"]
    (args.output_dir/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print((args.output_dir/"summary.md").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_coordinate_argument(p)
    p.add_argument("--ann-files", nargs="+", required=True, type=Path)
    p.add_argument("--bbox-root", type=Path, default=Path("data/K-RadarOcc/train"))
    p.add_argument("--bins", type=Path, default=Path("data/K-Radar-official-meta/resources/info_arr.mat"))
    p.add_argument("--extrinsic", type=Path)
    p.add_argument("--scenes", nargs="+", help="Optional exact scene IDs")
    p.add_argument("--start-order", type=int, default=80)
    p.add_argument("--end-order", type=int, default=200)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames-per-scene", type=int, default=0)
    p.add_argument("--strict-purity", type=float, default=.6)
    p.add_argument("--strict-min-points", type=int, default=3)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    if not (0 < args.strict_purity <= 1 and args.strict_min_points >= 1
            and args.stride > 0 and args.start_order <= args.end_order
            and args.max_frames_per_scene >= 0):
        p.error("Invalid score thresholds or sample window")
    execute(args)


if __name__ == "__main__":
    main()
