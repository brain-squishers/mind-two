import time

import cv2
import numpy as np
import torch

from live.memory_pipeline import apply_memory_fallback
from live.query_pipeline import advance_target_query_candidate
from spatial_reasoning import mask_to_xyxy


def handle_detection_error(detection_result, state):
    if detection_result is None or detection_result.get("error") is None:
        return detection_result

    print(
        f"{detection_result['job_type']} detect error: "
        f"{detection_result['error']}"
    )
    if detection_result["job_type"] == "target":
        state.tracking.pending_detection_request = None
        state.tracking.force_redetect = True
    elif detection_result["job_type"] == "anchor":
        state.context.pending_anchor_request = None
    elif detection_result["job_type"] == "support":
        state.context.pending_support_request = None
    elif detection_result["job_type"] == "hand":
        state.context.pending_hand_request = None
    return None


def handle_target_detection_result(detection_result, predictor, state, config):
    if detection_result is None or detection_result["job_type"] != "target":
        return False
    if detection_result["text"] != state.query.active_target_query:
        state.tracking.pending_detection_request = None
        return True

    result = detection_result["result"]
    boxes = result["boxes"]
    scores = result["scores"]
    print(
        f"target candidate '{state.query.active_target_query}' redetect boxes: "
        f"{boxes.shape[0]}"
    )

    if boxes.shape[0] != 0:
        keep_count = min(config.max_objects, boxes.shape[0])
        if keep_count < boxes.shape[0]:
            top_indices = torch.argsort(scores, descending=True)[:keep_count]
            boxes = boxes[top_indices]
        predictor.load_first_frame(detection_result["frame"])
        state.tracking.obj_display_names = {}
        for i, box in enumerate(boxes):
            obj_id = i + 1
            predictor.add_new_points(frame_idx=0, obj_id=obj_id, box=box)
            state.tracking.obj_display_names[obj_id] = (
                f"{state.query.current_label} #{obj_id}"
            )
        state.tracking.tracking_ready = True
        state.tracking.last_mask_rgb = None
        state.tracking.low_mask_count = 0
        state.tracking.tracking_steps = 0
        state.tracking.force_redetect = False
        state.memory.latest_memory_entry = None
        state.memory.latest_memory_response = ""
    else:
        if advance_target_query_candidate(state.query):
            state.tracking.force_redetect = True
            print("switching target candidate to:", state.query.active_target_query)
        else:
            state.tracking.force_redetect = True

    state.tracking.pending_detection_request = None
    return True


def maybe_schedule_target_detection(frame, grounding_worker, state, config):
    if not state.query.active_target_query:
        return False

    if not state.tracking.tracking_ready:
        should_redetect = (
            state.tracking.force_redetect
            or state.processed_frames == 1
            or (
                config.init_redetect_every > 0
                and state.processed_frames % config.init_redetect_every == 0
            )
        )
    else:
        should_redetect = (
            config.redetect_every > 0
            and state.processed_frames % config.redetect_every == 0
        )

    if not should_redetect:
        return False

    request_id = state.processed_frames
    if state.tracking.pending_detection_request == request_id:
        return False

    submitted = grounding_worker.submit(
        frame,
        state.query.active_target_query,
        request_id,
        job_type="target",
        box_threshold=config.target_box_threshold,
        text_threshold=config.target_text_threshold,
    )
    if not submitted:
        return False

    state.tracking.pending_detection_request = request_id
    state.tracking.force_redetect = False
    return True


def _clear_tracking_outputs(state):
    state.depth.latest_depth_stats = {}
    state.depth.latest_mask_ratio = 0.0
    state.depth.stable_mask_count = 0
    state.tracking.latest_label_positions = {}
    state.tracking.latest_target_boxes = []
    state.response.latest_relations = []
    state.response.latest_spatial_index = {}
    state.response.latest_scene_payload = None
    state.response.latest_spatial_response = ""
    state.response.pending_spatial_response = False
    state.response.last_scene_payload_key = None
    state.response.last_spatial_response_time = 0.0
    state.context.latest_anchor_detections = []
    state.context.latest_support_detections = []
    state.context.latest_hand_detections = []
    state.context.latest_anchor_frame_idx = -1
    state.context.latest_support_frame_idx = -1
    state.context.latest_hand_frame_idx = -1


def _handle_tracking_loss(scene_memory, state, config):
    state.tracking.tracking_ready = False
    state.tracking.last_mask_rgb = None
    state.tracking.low_mask_count = 0
    state.tracking.pending_detection_request = None
    state.tracking.force_redetect = True
    state.tracking.tracking_steps = 0
    _clear_tracking_outputs(state)
    apply_memory_fallback(scene_memory, state, max_age_s=30.0)
    print("tracking lost; returning to detection")


def _handle_empty_tracking_loss(scene_memory, state, config):
    state.tracking.tracking_ready = False
    state.tracking.low_mask_count = 0
    state.tracking.pending_detection_request = None
    state.context.pending_anchor_request = None
    state.context.pending_support_request = None
    state.tracking.force_redetect = True
    state.tracking.tracking_steps = 0
    _clear_tracking_outputs(state)
    apply_memory_fallback(scene_memory, state, max_age_s=30.0)
    print("tracking lost; returning to detection")


def run_tracking_step(
    frame,
    predictor,
    compute_mask_depth_stats,
    depth_kernel,
    scene_memory,
    state,
    config,
):
    state.tracking.tracking_steps += 1
    should_track = state.tracking.tracking_steps == 1 or (
        config.track_every > 0
        and state.tracking.tracking_steps % config.track_every == 0
    )
    if not should_track:
        return False

    out_obj_ids, out_mask_logits = predictor.track(frame)
    width, height = frame.shape[:2][::-1]
    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    current_depth_stats = {}
    current_label_positions = {}
    current_target_boxes = []

    for i in range(len(out_obj_ids)):
        obj_id = int(out_obj_ids[i])
        binary_mask = (
            (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        )
        out_mask = binary_mask[:, :, None] * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)
        target_box = mask_to_xyxy(binary_mask)
        if target_box is not None:
            current_target_boxes.append(target_box)
        ys, xs = np.nonzero(binary_mask)
        if xs.size != 0:
            current_label_positions[obj_id] = (int(xs.mean()), int(ys.mean()))
        if config.enable_depth and state.depth.latest_depth_map is not None:
            stats = compute_mask_depth_stats(
                state.depth.latest_depth_map,
                binary_mask,
                min_mask_pixels=config.depth_min_mask_pixels,
                erode_kernel=depth_kernel,
                max_depth=config.depth_max_depth,
            )
            if stats is not None:
                current_depth_stats[obj_id] = stats

    if len(out_obj_ids) != 0:
        state.depth.latest_depth_stats = current_depth_stats
        state.tracking.latest_label_positions = current_label_positions
        state.tracking.latest_target_boxes = current_target_boxes
        state.tracking.last_mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        mask_pixels = int(np.count_nonzero(all_mask))
        mask_ratio = mask_pixels / float(width * height)
        print(f"track mask_pixels: {mask_pixels} mask_ratio: {mask_ratio:.4f}")
        print("cache updated")
        state.depth.latest_mask_ratio = mask_ratio
        if mask_ratio >= config.depth_stable_mask_ratio:
            state.depth.stable_mask_count += 1
        else:
            state.depth.stable_mask_count = 0
        if mask_ratio < config.lost_mask_ratio:
            state.tracking.low_mask_count += 1
            print(f"weak track: {state.tracking.low_mask_count}/{config.lost_patience}")
        else:
            state.tracking.low_mask_count = 0

        if state.tracking.low_mask_count >= config.lost_patience:
            _handle_tracking_loss(scene_memory, state, config)
    else:
        state.tracking.last_mask_rgb = None
        _clear_tracking_outputs(state)
        state.tracking.low_mask_count += 1
        print("track produced no object ids; cache cleared")
        if state.tracking.low_mask_count >= config.lost_patience:
            _handle_empty_tracking_loss(scene_memory, state, config)

    return True
