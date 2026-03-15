import argparse
import asyncio
import collections
import re

import cv2
import numpy as np
import torch

from live_runtime import (
    DepthWorker,
    FpsTracker,
    GroundingWorker,
    LatestFrameCapture,
    compute_mask_depth_stats,
    device,
    extract_handler,
    load_depth_model,
    load_model,
)
from spatial_reasoning import (
    compute_iou_xyxy,
    compute_target_relations,
    mask_to_xyxy,
    select_top_relations,
)
from utils import add_text_with_background


print("device", device)


def format_query_label(text):
    label = re.sub(r"[\s\.,;:!?]+$", "", text.strip())
    return label or "object"


def format_query_summary(
    targets,
    anchors,
    support_surfaces,
    active_target_query=None,
):
    parts = []
    if active_target_query:
        parts.append(f"target: {active_target_query}")
    elif targets:
        parts.append(f"target: {targets[0]}")
    if anchors:
        parts.append(f"anchors: {', '.join(anchors[:3])}")
    if support_surfaces:
        parts.append(f"surfaces: {', '.join(support_surfaces[:3])}")
    return " | ".join(parts)


def build_phrase_query(phrases):
    return ". ".join(phrase for phrase in phrases if phrase)


def get_active_target_query(target_queries, target_query_index):
    if not target_queries:
        return ""
    if target_query_index < 0 or target_query_index >= len(target_queries):
        return ""
    return target_queries[target_query_index]


def normalize_phrase_key(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def choose_detection_label(raw_label, fallback_phrases):
    cleaned_label = str(raw_label).strip()
    if not fallback_phrases:
        return cleaned_label or "object"

    normalized_label = normalize_phrase_key(cleaned_label)
    if normalized_label:
        for phrase in fallback_phrases:
            normalized_phrase = normalize_phrase_key(phrase)
            if not normalized_phrase:
                continue
            if (
                normalized_phrase in normalized_label
                or normalized_label in normalized_phrase
            ):
                return phrase

        label_tokens = set(normalized_label.split())
        best_phrase = ""
        best_overlap = 0
        for phrase in fallback_phrases:
            phrase_tokens = set(normalize_phrase_key(phrase).split())
            overlap = len(label_tokens & phrase_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_phrase = phrase
        if best_phrase:
            return best_phrase

    return cleaned_label or fallback_phrases[0]


def suppress_target_overlaps(
    detections,
    target_boxes,
    target_queries,
    iou_threshold=0.2,
):
    if not detections or not target_boxes:
        return detections

    normalized_targets = {
        normalize_phrase_key(target_query)
        for target_query in target_queries
        if normalize_phrase_key(target_query)
    }

    filtered = []
    for detection in detections:
        detection_label_key = normalize_phrase_key(detection["label"])
        if detection_label_key in normalized_targets:
            continue

        overlaps_target = False
        for target_box in target_boxes:
            if compute_iou_xyxy(detection["box"], target_box) >= iou_threshold:
                overlaps_target = True
                break

        if not overlaps_target:
            filtered.append(detection)

    return filtered


def extract_labeled_boxes(result, fallback_phrases):
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result.get("labels")

    detections = []
    for i in range(boxes.shape[0]):
        raw_label = ""
        if labels is not None and i < len(labels):
            raw_label = labels[i]
        label = choose_detection_label(raw_label, fallback_phrases)

        detections.append(
            {
                "label": label,
                "score": float(scores[i]),
                "box": boxes[i].detach().cpu().numpy(),
            }
        )

    return detections


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


async def main(
    model="gpt-4o-2024-05-13",
    camera_index=0,
    skip_frames=2,
    track_every=3,
    init_redetect_every=10,
    redetect_every=30,
    lost_mask_ratio=0.001,
    lost_patience=5,
    max_objects=3,
    box_threshold=0.35,
    text_threshold=0.25,
    target_box_threshold=0.30,
    target_text_threshold=0.25,
    fallback_target_box_threshold=0.22,
    fallback_target_text_threshold=0.18,
    query="I am trying to find my glass",
    enable_depth=True,
    depth_encoder="vits",
    depth_dataset="hypersim",
    depth_max_depth=20.0,
    depth_checkpoint=None,
    depth_every=10,
    depth_input_size=336,
    depth_min_mask_pixels=300,
    depth_mask_erode_kernel=5,
    depth_stable_mask_ratio=0.01,
    depth_stable_patience=2,
    anchor_redetect_every=45,
    support_redetect_every=75,
    anchor_cache_ttl=60,
    support_cache_ttl=90,
):
    grounding_processor, grounding_model, predictor, llm = load_model(model)
    if enable_depth:
        depth_model, depth_checkpoint_path = load_depth_model(
            depth_encoder=depth_encoder,
            depth_dataset=depth_dataset,
            depth_max_depth=depth_max_depth,
            depth_checkpoint=depth_checkpoint,
        )
        print(f"depth checkpoint {depth_checkpoint_path}")
        depth_worker = DepthWorker(
            depth_model,
            input_size=depth_input_size,
        ).start()
    else:
        depth_worker = None

    capture = LatestFrameCapture(camera_index).start()
    grounding_worker = GroundingWorker(
        grounding_processor,
        grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    ).start()

    query_queue = collections.deque([query])
    response_queue = collections.deque([])

    current_query = query
    current_query_summary = query
    target_queries = []
    target_query_index = 0
    target_fallback_active = False
    anchor_queries = []
    support_surface_queries = []
    active_target_query = ""
    active_anchor_query = ""
    active_support_query = ""
    current_label = "object"

    tracking_ready = False
    last_mask_rgb = None
    low_mask_count = 0
    tracking_steps = 0

    latest_depth_map = None
    latest_depth_frame_idx = -1
    latest_depth_stats = {}
    pending_depth_request = None
    latest_mask_ratio = 0.0
    stable_mask_count = 0
    obj_display_names = {}
    latest_label_positions = {}
    latest_target_boxes = []
    latest_anchor_detections = []
    latest_support_detections = []
    latest_relations = []
    latest_anchor_frame_idx = -1
    latest_support_frame_idx = -1

    frame_count = 0
    processed_frames = 0
    pending_detection_request = None
    pending_anchor_request = None
    pending_support_request = None
    force_redetect = True
    fps_tracker = FpsTracker()

    depth_kernel = None
    if depth_mask_erode_kernel > 1:
        depth_kernel = np.ones(
            (depth_mask_erode_kernel, depth_mask_erode_kernel), dtype=np.uint8
        )

    try:
        while True:
            frame = capture.read_latest()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            frame_count += 1
            display_frame = frame.copy()

            if depth_worker is not None:
                depth_result = depth_worker.poll_result()
                if depth_result is not None:
                    if depth_result.get("error") is not None:
                        print(f"depth error: {depth_result['error']}")
                    else:
                        latest_depth_map = depth_result["depth_map"]
                        latest_depth_frame_idx = depth_result["request_id"]
                    pending_depth_request = None

            detection_result = grounding_worker.poll_result()
            if detection_result is not None:
                if detection_result.get("error") is not None:
                    print(
                        f"{detection_result['job_type']} detect error: "
                        f"{detection_result['error']}"
                    )
                    if detection_result["job_type"] == "target":
                        pending_detection_request = None
                        force_redetect = True
                    elif detection_result["job_type"] == "anchor":
                        pending_anchor_request = None
                    elif detection_result["job_type"] == "support":
                        pending_support_request = None
                    detection_result = None
                if detection_result is not None:
                    job_type = detection_result["job_type"]
                    result = detection_result["result"]

                    if job_type == "target":
                        if detection_result["text"] == active_target_query:
                            boxes = result["boxes"]
                            scores = result["scores"]
                            print(
                                f"target candidate '{active_target_query}' "
                                f"redetect boxes: {boxes.shape[0]}"
                            )
                            if boxes.shape[0] != 0:
                                keep_count = min(max_objects, boxes.shape[0])
                                if keep_count < boxes.shape[0]:
                                    top_indices = torch.argsort(
                                        scores, descending=True
                                    )[:keep_count]
                                    boxes = boxes[top_indices]
                                predictor.load_first_frame(detection_result["frame"])
                                obj_display_names = {}
                                for i, box in enumerate(boxes):
                                    obj_id = i + 1
                                    predictor.add_new_points(
                                        frame_idx=0,
                                        obj_id=obj_id,
                                        box=box,
                                    )
                                    obj_display_names[obj_id] = (
                                        f"{current_label} #{obj_id}"
                                    )
                                tracking_ready = True
                                last_mask_rgb = None
                                low_mask_count = 0
                                tracking_steps = 0
                                force_redetect = False
                                target_fallback_active = False
                            else:
                                if target_query_index + 1 < len(target_queries):
                                    target_query_index += 1
                                    active_target_query = get_active_target_query(
                                        target_queries,
                                        target_query_index,
                                    )
                                    current_label = format_query_label(
                                        active_target_query
                                    )
                                    current_query_summary = format_query_summary(
                                        target_queries,
                                        anchor_queries,
                                        support_surface_queries,
                                        active_target_query=active_target_query,
                                    )
                                    force_redetect = True
                                    print(
                                        "switching target candidate to:",
                                        active_target_query,
                                    )
                                elif not target_fallback_active:
                                    target_fallback_active = True
                                    target_query_index = 0
                                    active_target_query = get_active_target_query(
                                        target_queries,
                                        target_query_index,
                                    )
                                    current_label = format_query_label(
                                        active_target_query
                                    )
                                    current_query_summary = format_query_summary(
                                        target_queries,
                                        anchor_queries,
                                        support_surface_queries,
                                        active_target_query=active_target_query,
                                    )
                                    force_redetect = True
                                    print(
                                        "switching target search to fallback "
                                        f"thresholds with candidate: {active_target_query}"
                                    )
                                else:
                                    force_redetect = True
                        pending_detection_request = None
                    elif job_type == "anchor":
                        if detection_result["text"] == active_anchor_query:
                            latest_anchor_detections = suppress_target_overlaps(
                                extract_labeled_boxes(
                                    result,
                                    anchor_queries,
                                ),
                                latest_target_boxes,
                                target_queries,
                            )
                            latest_anchor_frame_idx = processed_frames
                        pending_anchor_request = None
                    elif job_type == "support":
                        if detection_result["text"] == active_support_query:
                            latest_support_detections = suppress_target_overlaps(
                                extract_labeled_boxes(
                                    result,
                                    support_surface_queries,
                                ),
                                latest_target_boxes,
                                target_queries,
                            )
                            latest_support_frame_idx = processed_frames
                        pending_support_request = None

            if query_queue:
                pending_query = query_queue.popleft()
                asyncio.create_task(extract_handler(pending_query, response_queue, llm))

            if response_queue:
                extraction = response_queue.popleft()
                target_queries = extraction["targets"]
                target_query_index = 0
                target_fallback_active = False
                anchor_queries = extraction["anchors"]
                support_surface_queries = extraction["support_surfaces"]
                active_target_query = get_active_target_query(
                    target_queries,
                    target_query_index,
                )
                active_anchor_query = build_phrase_query(anchor_queries)
                active_support_query = build_phrase_query(support_surface_queries)
                current_label = format_query_label(active_target_query)
                current_query_summary = format_query_summary(
                    target_queries,
                    anchor_queries,
                    support_surface_queries,
                    active_target_query=active_target_query,
                )
                print(
                    "extraction:",
                    {
                        "targets": target_queries,
                        "anchors": anchor_queries,
                        "support_surfaces": support_surface_queries,
                    },
                )
                tracking_ready = False
                last_mask_rgb = None
                low_mask_count = 0
                tracking_steps = 0
                pending_detection_request = None
                pending_anchor_request = None
                pending_support_request = None
                force_redetect = True
                latest_depth_map = None
                latest_depth_frame_idx = -1
                latest_depth_stats = {}
                pending_depth_request = None
                latest_mask_ratio = 0.0
                stable_mask_count = 0
                obj_display_names = {}
                latest_label_positions = {}
                latest_target_boxes = []
                latest_anchor_detections = []
                latest_support_detections = []
                latest_relations = []
                latest_anchor_frame_idx = -1
                latest_support_frame_idx = -1

            should_process = frame_count == 1 or frame_count % skip_frames == 0

            if active_target_query and should_process:
                processed_frames += 1
                if (
                    latest_anchor_frame_idx >= 0
                    and anchor_cache_ttl > 0
                    and processed_frames - latest_anchor_frame_idx > anchor_cache_ttl
                ):
                    latest_anchor_detections = []
                    latest_anchor_frame_idx = -1

                if (
                    latest_support_frame_idx >= 0
                    and support_cache_ttl > 0
                    and processed_frames - latest_support_frame_idx > support_cache_ttl
                ):
                    latest_support_detections = []
                    latest_support_frame_idx = -1

                if not tracking_ready:
                    should_redetect = (
                        force_redetect
                        or processed_frames == 1
                        or (
                            init_redetect_every > 0
                            and processed_frames % init_redetect_every == 0
                        )
                    )
                else:
                    should_redetect = (
                        redetect_every > 0 and processed_frames % redetect_every == 0
                    )

                if should_redetect:
                    request_id = processed_frames
                    if pending_detection_request != request_id:
                        submitted = grounding_worker.submit(
                            frame,
                            active_target_query,
                            request_id,
                            job_type="target",
                            box_threshold=(
                                fallback_target_box_threshold
                                if target_fallback_active
                                else target_box_threshold
                            ),
                            text_threshold=(
                                fallback_target_text_threshold
                                if target_fallback_active
                                else target_text_threshold
                            ),
                        )
                        if submitted:
                            pending_detection_request = request_id
                            force_redetect = False
                elif tracking_ready:
                    depth_tracking_stable = (
                        last_mask_rgb is not None
                        and latest_mask_ratio >= depth_stable_mask_ratio
                        and stable_mask_count >= depth_stable_patience
                    )
                    should_run_depth = enable_depth and (
                        latest_depth_map is None
                        or (depth_tracking_stable and tracking_steps == 0)
                        or (
                            depth_tracking_stable
                            and depth_every > 0
                            and processed_frames % depth_every == 0
                        )
                    )
                    if should_run_depth and depth_worker is not None:
                        request_id = processed_frames
                        if pending_depth_request != request_id:
                            depth_worker.submit(frame, request_id)
                            pending_depth_request = request_id

                    tracking_steps += 1
                    should_track = tracking_steps == 1 or (
                        track_every > 0 and tracking_steps % track_every == 0
                    )
                    if should_track:
                        out_obj_ids, out_mask_logits = predictor.track(frame)
                        width, height = frame.shape[:2][::-1]
                        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                        current_depth_stats = {}
                        current_label_positions = {}
                        current_target_boxes = []
                        for i in range(len(out_obj_ids)):
                            obj_id = int(out_obj_ids[i])
                            binary_mask = (
                                (out_mask_logits[i] > 0.0)
                                .squeeze(0)
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            out_mask = binary_mask[:, :, None] * 255
                            all_mask = cv2.bitwise_or(all_mask, out_mask)
                            target_box = mask_to_xyxy(binary_mask)
                            if target_box is not None:
                                current_target_boxes.append(target_box)
                            ys, xs = np.nonzero(binary_mask)
                            if xs.size != 0:
                                current_label_positions[obj_id] = (
                                    int(xs.mean()),
                                    int(ys.mean()),
                                )
                            if enable_depth and latest_depth_map is not None:
                                stats = compute_mask_depth_stats(
                                    latest_depth_map,
                                    binary_mask,
                                    min_mask_pixels=depth_min_mask_pixels,
                                    erode_kernel=depth_kernel,
                                    max_depth=depth_max_depth,
                                )
                                if stats is not None:
                                    current_depth_stats[obj_id] = stats

                        if len(out_obj_ids) != 0:
                            latest_depth_stats = current_depth_stats
                            latest_label_positions = current_label_positions
                            latest_target_boxes = current_target_boxes
                            last_mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                            mask_pixels = int(np.count_nonzero(all_mask))
                            mask_ratio = mask_pixels / float(width * height)
                            print(
                                f"track mask_pixels: {mask_pixels} mask_ratio: {mask_ratio:.4f}"
                            )
                            print("cache updated")
                            latest_mask_ratio = mask_ratio
                            if mask_ratio >= depth_stable_mask_ratio:
                                stable_mask_count += 1
                            else:
                                stable_mask_count = 0
                            if mask_ratio < lost_mask_ratio:
                                low_mask_count += 1
                                print(f"weak track: {low_mask_count}/{lost_patience}")
                            else:
                                low_mask_count = 0

                            if low_mask_count >= lost_patience:
                                tracking_ready = False
                                last_mask_rgb = None
                                low_mask_count = 0
                                pending_detection_request = None
                                force_redetect = True
                                tracking_steps = 0
                                latest_depth_stats = {}
                                latest_mask_ratio = 0.0
                                stable_mask_count = 0
                                latest_label_positions = {}
                                latest_target_boxes = []
                                latest_relations = []
                                latest_anchor_detections = []
                                latest_support_detections = []
                                latest_anchor_frame_idx = -1
                                latest_support_frame_idx = -1
                                print("tracking lost; returning to detection")
                        else:
                            last_mask_rgb = None
                            latest_depth_stats = {}
                            latest_mask_ratio = 0.0
                            stable_mask_count = 0
                            latest_label_positions = {}
                            latest_target_boxes = []
                            latest_relations = []
                            latest_anchor_detections = []
                            latest_support_detections = []
                            latest_anchor_frame_idx = -1
                            latest_support_frame_idx = -1
                            low_mask_count += 1
                            print("track produced no object ids; cache cleared")
                            if low_mask_count >= lost_patience:
                                tracking_ready = False
                                low_mask_count = 0
                                pending_detection_request = None
                                pending_anchor_request = None
                                pending_support_request = None
                                force_redetect = True
                                tracking_steps = 0
                                latest_depth_stats = {}
                                latest_mask_ratio = 0.0
                                stable_mask_count = 0
                                latest_label_positions = {}
                                latest_target_boxes = []
                                latest_relations = []
                                latest_anchor_detections = []
                                latest_support_detections = []
                                latest_anchor_frame_idx = -1
                                latest_support_frame_idx = -1
                                print("tracking lost; returning to detection")

                can_schedule_context = (
                    pending_detection_request is None
                    and active_target_query
                    and tracking_ready
                )

                should_refresh_anchors = (
                    can_schedule_context
                    and active_anchor_query
                    and anchor_redetect_every > 0
                    and processed_frames % anchor_redetect_every == 0
                )
                if should_refresh_anchors:
                    request_id = processed_frames
                    if pending_anchor_request != request_id:
                        submitted = grounding_worker.submit(
                            frame,
                            active_anchor_query,
                            request_id,
                            job_type="anchor",
                        )
                        if submitted:
                            pending_anchor_request = request_id

                should_refresh_support = (
                    can_schedule_context
                    and active_support_query
                    and support_redetect_every > 0
                    and processed_frames % support_redetect_every == 0
                )
                if should_refresh_support and pending_anchor_request is None:
                    request_id = processed_frames
                    if pending_support_request != request_id:
                        submitted = grounding_worker.submit(
                            frame,
                            active_support_query,
                            request_id,
                            job_type="support",
                        )
                        if submitted:
                            pending_support_request = request_id

                latest_relations = compute_target_relations(
                    frame.shape,
                    current_label,
                    latest_target_boxes,
                    latest_depth_stats,
                    latest_anchor_detections,
                    latest_support_detections,
                    depth_map=latest_depth_map,
                    max_depth=depth_max_depth,
                )

            if last_mask_rgb is not None:
                display_frame = cv2.addWeighted(display_frame, 1, last_mask_rgb, 0.5, 0)

            if current_query_summary:
                display_frame = add_text_with_background(
                    display_frame,
                    current_query_summary,
                )

            fps = fps_tracker.tick()
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
            status = "TRACKING" if tracking_ready else "DETECTING"
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
                    f"DEPTH cached {latest_depth_frame_idx}"
                    if latest_depth_map is not None
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
                for obj_id in sorted(latest_depth_stats):
                    median_m = latest_depth_stats[obj_id]["median_m"]
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
            for obj_id, position in latest_label_positions.items():
                label = obj_display_names.get(obj_id, f"{current_label} #{obj_id}")
                depth_suffix = ""
                if obj_id in latest_depth_stats:
                    depth_suffix = f" {latest_depth_stats[obj_id]['median_m']:.1f} m"
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
            draw_box_detections(display_frame, latest_anchor_detections, (0, 200, 255))
            draw_box_detections(display_frame, latest_support_detections, (255, 0, 200))
            relation_y = 165
            for relation in select_top_relations(latest_relations):
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

            cv2.imshow("run_live", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            await asyncio.sleep(0)

    finally:
        if depth_worker is not None:
            depth_worker.release()
        grounding_worker.release()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-latency live webcam runner.")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--skip-frames", type=int, default=2)
    parser.add_argument("--track-every", type=int, default=3)
    parser.add_argument("--init-redetect-every", type=int, default=10)
    parser.add_argument("--redetect-every", type=int, default=30)
    parser.add_argument("--lost-mask-ratio", type=float, default=0.001)
    parser.add_argument("--lost-patience", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--target-box-threshold", type=float, default=0.30)
    parser.add_argument("--target-text-threshold", type=float, default=0.25)
    parser.add_argument("--fallback-target-box-threshold", type=float, default=0.22)
    parser.add_argument("--fallback-target-text-threshold", type=float, default=0.18)
    parser.add_argument(
        "--disable-depth",
        action="store_false",
        dest="enable_depth",
        help="Disable metric depth estimation in the live pipeline.",
    )
    parser.set_defaults(enable_depth=True)
    parser.add_argument(
        "--depth-encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl"],
    )
    parser.add_argument(
        "--depth-dataset",
        type=str,
        default="hypersim",
        choices=["hypersim", "vkitti"],
    )
    parser.add_argument("--depth-max-depth", type=float, default=20.0)
    parser.add_argument("--depth-checkpoint", type=str, default=None)
    parser.add_argument("--depth-every", type=int, default=10)
    parser.add_argument("--depth-input-size", type=int, default=336)
    parser.add_argument("--depth-min-mask-pixels", type=int, default=300)
    parser.add_argument("--depth-mask-erode-kernel", type=int, default=5)
    parser.add_argument("--depth-stable-mask-ratio", type=float, default=0.01)
    parser.add_argument("--depth-stable-patience", type=int, default=2)
    parser.add_argument("--anchor-redetect-every", type=int, default=45)
    parser.add_argument("--support-redetect-every", type=int, default=75)
    parser.add_argument("--anchor-cache-ttl", type=int, default=60)
    parser.add_argument("--support-cache-ttl", type=int, default=90)
    parser.add_argument("--query", type=str, default="I am trying to find phones")
    # parser.add_argument(
    #     "--query", type=str, default="I cannot see well since i hv myopia. help"
    # )
    # parser.add_argument("--query", type=str, default="I am thirsty. where is my flask?")
    args = parser.parse_args()

    asyncio.run(
        main(
            model=args.model,
            camera_index=args.camera_index,
            skip_frames=args.skip_frames,
            track_every=args.track_every,
            init_redetect_every=args.init_redetect_every,
            redetect_every=args.redetect_every,
            lost_mask_ratio=args.lost_mask_ratio,
            lost_patience=args.lost_patience,
            max_objects=args.max_objects,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_box_threshold=args.target_box_threshold,
            target_text_threshold=args.target_text_threshold,
            fallback_target_box_threshold=args.fallback_target_box_threshold,
            fallback_target_text_threshold=args.fallback_target_text_threshold,
            query=args.query,
            enable_depth=args.enable_depth,
            depth_encoder=args.depth_encoder,
            depth_dataset=args.depth_dataset,
            depth_max_depth=args.depth_max_depth,
            depth_checkpoint=args.depth_checkpoint,
            depth_every=args.depth_every,
            depth_input_size=args.depth_input_size,
            depth_min_mask_pixels=args.depth_min_mask_pixels,
            depth_mask_erode_kernel=args.depth_mask_erode_kernel,
            depth_stable_mask_ratio=args.depth_stable_mask_ratio,
            depth_stable_patience=args.depth_stable_patience,
            anchor_redetect_every=args.anchor_redetect_every,
            support_redetect_every=args.support_redetect_every,
            anchor_cache_ttl=args.anchor_cache_ttl,
            support_cache_ttl=args.support_cache_ttl,
        )
    )
