import cv2

from spatial_reasoning import select_top_relations
from utils import add_text_with_background


def draw_box_detections(frame, detections, color):
    overlay = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"].astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"].astype(int)
        cv2.putText(
            frame,
            f"{detection['label']} {detection['score']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def format_relation_text(relation):
    relation_text = relation["type"].replace("_", " ")
    if relation["type"] in {"on_your_left", "on_your_right", "straight_ahead"}:
        return f"{relation['subject']} {relation_text}"
    return f"{relation['subject']} {relation_text} {relation['object']}"


def render_display_frame(
    frame,
    *,
    query_state,
    tracking_state,
    depth_state,
    context_state,
    response_state,
    memory_state,
    fps,
    enable_depth,
):
    display_frame = frame.copy()

    if tracking_state.last_mask_rgb is not None:
        display_frame = cv2.addWeighted(
            display_frame,
            1,
            tracking_state.last_mask_rgb,
            0.5,
            0,
        )

    if query_state.current_query_summary:
        display_frame = add_text_with_background(
            display_frame,
            query_state.current_query_summary,
        )

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    status = "TRACKING" if tracking_state.tracking_ready else "DETECTING"
    cv2.putText(
        display_frame,
        status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if enable_depth:
        depth_status = (
            f"DEPTH cached {depth_state.latest_depth_frame_idx}"
            if depth_state.latest_depth_map is not None
            else "DEPTH waiting"
        )
        cv2.putText(
            display_frame,
            depth_status,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        y = 135
        for obj_id in sorted(depth_state.latest_depth_stats):
            median_m = depth_state.latest_depth_stats[obj_id]["median_m"]
            cv2.putText(
                display_frame,
                f"obj {obj_id}: {median_m:.2f} m",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28

    for obj_id, position in tracking_state.latest_label_positions.items():
        label = tracking_state.obj_display_names.get(
            obj_id,
            f"{query_state.current_label} #{obj_id}",
        )
        depth_suffix = ""
        if obj_id in depth_state.latest_depth_stats:
            depth_suffix = (
                f" {depth_state.latest_depth_stats[obj_id]['median_m']:.1f} m"
            )
        cv2.putText(
            display_frame,
            f"{label}{depth_suffix}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    draw_box_detections(display_frame, context_state.latest_anchor_detections, (0, 200, 255))
    draw_box_detections(display_frame, context_state.latest_support_detections, (255, 0, 200))

    target_entry = response_state.latest_spatial_index.get("target_1")
    if target_entry is not None:
        x_m, y_m, z_m = target_entry["position_3d_m"]
        depth_std_m = target_entry["depth_std_m"]
        xyz_text = f"xyz {x_m:.2f}, {y_m:.2f}, {z_m:.2f} m"
        if depth_std_m is not None:
            xyz_text += f" +- {depth_std_m:.2f}"
        cv2.putText(
            display_frame,
            xyz_text,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )

    draw_box_detections(display_frame, context_state.latest_hand_detections, (80, 255, 80))

    relation_y = 165
    for relation in select_top_relations(response_state.latest_relations):
        cv2.putText(
            display_frame,
            format_relation_text(relation),
            (20, relation_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )
        relation_y += 26

    if response_state.latest_spatial_response:
        cv2.putText(
            display_frame,
            response_state.latest_spatial_response,
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if (not tracking_state.tracking_ready) and memory_state.latest_memory_response:
        cv2.putText(
            display_frame,
            memory_state.latest_memory_response,
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 180),
            2,
            cv2.LINE_AA,
        )

    return display_frame
