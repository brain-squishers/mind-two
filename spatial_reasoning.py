import numpy as np


def box_center_xy(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def box_size_xy(box):
    x1, y1, x2, y2 = box
    return np.array([max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)


def compute_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def mask_to_xyxy(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def compute_relative_offsets(target_box, other_box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    target_center = box_center_xy(target_box)
    other_center = box_center_xy(other_box)

    dx_px = float(other_center[0] - target_center[0])
    dy_px = float(other_center[1] - target_center[1])
    dx_norm = dx_px / max(1.0, float(frame_w))
    dy_norm = dy_px / max(1.0, float(frame_h))
    xy_distance_px = float(np.linalg.norm(other_center - target_center))
    xy_distance_norm = float(
        np.linalg.norm(
            np.array(
                [
                    dx_px / max(1.0, float(frame_w)),
                    dy_px / max(1.0, float(frame_h)),
                ],
                dtype=np.float32,
            )
        )
    )

    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "dx_norm": dx_norm,
        "dy_norm": dy_norm,
        "xy_distance_px": xy_distance_px,
        "xy_distance_norm": xy_distance_norm,
    }


def compute_box_depth(depth_map, box, max_depth=None):
    x1, y1, x2, y2 = box.astype(int)
    h, w = depth_map.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    values = depth_map[y1:y2, x1:x2].reshape(-1)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if max_depth is not None:
        values = values[values < max_depth]
    if values.size == 0:
        return None

    return float(np.median(values))


def compute_depth_relation(target_depth_m, other_depth_m, depth_margin_m=0.25):
    if target_depth_m is None or other_depth_m is None:
        return None

    delta_m = float(other_depth_m - target_depth_m)
    if delta_m > depth_margin_m:
        relation = "in_front_of"
    elif delta_m < -depth_margin_m:
        relation = "behind"
    else:
        relation = None

    return {
        "relation": relation,
        "delta_m": delta_m,
        "abs_delta_m": abs(delta_m),
    }


def compute_near_score(offsets, depth_delta_m=None):
    xy_score = max(0.0, 1.0 - offsets["xy_distance_norm"] / 0.35)
    if depth_delta_m is None:
        return xy_score

    depth_score = max(0.0, 1.0 - abs(depth_delta_m) / 0.75)
    return 0.6 * xy_score + 0.4 * depth_score


def compute_on_relation(
    target_box,
    support_box,
    target_depth_m=None,
    support_depth_m=None,
):
    tx1, ty1, tx2, ty2 = target_box
    sx1, sy1, sx2, sy2 = support_box

    target_w = max(1.0, tx2 - tx1)
    target_h = max(1.0, ty2 - ty1)
    horizontal_overlap = max(0.0, min(tx2, sx2) - max(tx1, sx1))
    horizontal_overlap_ratio = horizontal_overlap / target_w

    target_bottom = ty2
    support_top = sy1
    vertical_gap = target_bottom - support_top
    target_center = box_center_xy(target_box)
    center_inside_support = (
        sx1 <= target_center[0] <= sx2 and sy1 <= target_center[1] <= sy2
    )

    depth_ok = True
    depth_delta_m = None
    if target_depth_m is not None and support_depth_m is not None:
        depth_delta_m = support_depth_m - target_depth_m
        depth_ok = abs(depth_delta_m) <= 0.75

    vertically_plausible = (-0.20 * target_h) <= vertical_gap <= (0.35 * target_h)
    is_on = (
        horizontal_overlap_ratio >= 0.35
        and (vertically_plausible or center_inside_support)
        and depth_ok
    )

    score = 0.0
    if is_on:
        overlap_score = min(1.0, horizontal_overlap_ratio)
        gap_score = max(0.0, 1.0 - abs(vertical_gap) / max(1.0, target_h))
        depth_score = 1.0
        if depth_delta_m is not None:
            depth_score = max(0.0, 1.0 - abs(depth_delta_m) / 0.75)
        score = 0.5 * overlap_score + 0.3 * gap_score + 0.2 * depth_score

    return {
        "is_on": is_on,
        "score": float(score),
        "horizontal_overlap_ratio": float(horizontal_overlap_ratio),
        "vertical_gap_px": float(vertical_gap),
        "depth_delta_m": depth_delta_m,
    }


def get_primary_target_depth_m(target_depth_stats):
    if not target_depth_stats:
        return None
    first_obj_id = sorted(target_depth_stats)[0]
    return target_depth_stats[first_obj_id]["median_m"]


def compute_ego_relations(frame_shape, target_label, target_box, target_depth_m=None):
    frame_h, frame_w = frame_shape[:2]
    target_center = box_center_xy(target_box)
    x_offset_norm = (float(target_center[0]) - 0.5 * float(frame_w)) / max(
        1.0, 0.5 * float(frame_w)
    )

    if x_offset_norm <= -0.18:
        relation_type = "on_your_left"
    elif x_offset_norm >= 0.18:
        relation_type = "on_your_right"
    else:
        relation_type = "straight_ahead"

    score = max(0.0, 1.0 - abs(x_offset_norm))
    return [
        {
            "type": relation_type,
            "subject": target_label,
            "object": "you",
            "score": score,
            "metrics": {
                "x_offset_norm": x_offset_norm,
                "target_depth_m": target_depth_m,
            },
        }
    ]


def compute_target_relations(
    frame_shape,
    target_label,
    target_boxes,
    target_depth_stats,
    anchor_detections,
    support_detections,
    depth_map=None,
    max_depth=None,
):
    if not target_boxes:
        return []

    target_box = target_boxes[0]
    target_depth_m = get_primary_target_depth_m(target_depth_stats)
    relations = compute_ego_relations(
        frame_shape,
        target_label,
        target_box,
        target_depth_m=target_depth_m,
    )

    for detection in anchor_detections:
        other_box = detection["box"]
        offsets = compute_relative_offsets(target_box, other_box, frame_shape)
        horizontal_relation = "right_of" if offsets["dx_px"] > 0 else "left_of"

        other_depth_m = None
        if depth_map is not None:
            other_depth_m = compute_box_depth(
                depth_map,
                other_box,
                max_depth=max_depth,
            )

        depth_info = compute_depth_relation(target_depth_m, other_depth_m)
        near_score = compute_near_score(
            offsets,
            None if depth_info is None else depth_info["delta_m"],
        )

        horizontal_score = max(0.0, 1.0 - abs(offsets["dx_norm"]) / 0.5)

        if abs(offsets["dx_norm"]) >= 0.08:
            relations.append(
                {
                    "type": horizontal_relation,
                    "subject": target_label,
                    "object": detection["label"],
                    "score": horizontal_score,
                    "metrics": offsets,
                }
            )

        if depth_info is not None and depth_info["relation"] is not None:
            relations.append(
                {
                    "type": depth_info["relation"],
                    "subject": target_label,
                    "object": detection["label"],
                    "score": max(0.0, 1.0 - depth_info["abs_delta_m"] / 1.5),
                    "metrics": {
                        **offsets,
                        "depth_delta_m": depth_info["delta_m"],
                    },
                }
            )

        if near_score >= 0.45:
            relations.append(
                {
                    "type": "near",
                    "subject": target_label,
                    "object": detection["label"],
                    "score": near_score,
                    "metrics": {
                        **offsets,
                        "other_depth_m": other_depth_m,
                        "target_depth_m": target_depth_m,
                        "depth_delta_m": None if depth_info is None else depth_info["delta_m"],
                    },
                }
            )

    for detection in support_detections:
        support_box = detection["box"]
        support_depth_m = None
        if depth_map is not None:
            support_depth_m = compute_box_depth(depth_map, support_box, max_depth=max_depth)

        on_info = compute_on_relation(
            target_box,
            support_box,
            target_depth_m=target_depth_m,
            support_depth_m=support_depth_m,
        )
        if on_info["is_on"]:
            relations.append(
                {
                    "type": "on",
                    "subject": target_label,
                    "object": detection["label"],
                    "score": on_info["score"],
                    "metrics": {
                        "target_depth_m": target_depth_m,
                        "support_depth_m": support_depth_m,
                        "depth_delta_m": on_info["depth_delta_m"],
                        "horizontal_overlap_ratio": on_info["horizontal_overlap_ratio"],
                        "vertical_gap_px": on_info["vertical_gap_px"],
                    },
                }
            )

    return relations


def select_top_relations(relations, max_relations=3):
    priority = {
        "on_your_left": 0,
        "on_your_right": 0,
        "straight_ahead": 0,
        "on": 1,
        "near": 2,
        "left_of": 3,
        "right_of": 3,
        "in_front_of": 4,
        "behind": 4,
    }
    return sorted(
        relations,
        key=lambda rel: (priority.get(rel["type"], 99), -rel["score"]),
    )[:max_relations]
