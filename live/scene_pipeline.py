import asyncio
import json
import time

from scene_output import build_scene_output_payload
from spatial_reasoning import (
    build_spatial_index,
    compute_target_relations,
    select_top_relations,
)


def poll_spatial_response(state, output_sink):
    if not state.spatial_response_queue:
        return None

    state.response.latest_spatial_response = state.spatial_response_queue.popleft()
    state.response.pending_spatial_response = False
    print("spatial response:", state.response.latest_spatial_response)
    output_sink.publish_text(state.response.latest_spatial_response)
    return state.response.latest_spatial_response


def update_scene_state(state, config, frame_shape):
    state.response.latest_relations = compute_target_relations(
        frame_shape,
        state.query.current_label,
        state.tracking.latest_target_boxes,
        state.depth.latest_depth_stats,
        state.context.latest_anchor_detections,
        state.context.latest_support_detections,
        state.context.latest_hand_detections,
        depth_map=state.depth.latest_depth_map,
        max_depth=config.depth_max_depth,
    )
    state.response.latest_spatial_index = build_spatial_index(
        frame_shape,
        state.query.current_label,
        state.tracking.latest_target_boxes,
        state.depth.latest_depth_stats,
        state.context.latest_anchor_detections,
        state.context.latest_support_detections,
        depth_map=state.depth.latest_depth_map,
        max_depth=config.depth_max_depth,
    )
    state.response.latest_scene_payload = build_scene_output_payload(
        state.query.current_query,
        state.query.current_label,
        state.response.latest_spatial_index,
        select_top_relations(state.response.latest_relations),
    )
    return state.response.latest_scene_payload


def maybe_schedule_spatial_response(state, config, llm, spatial_response_handler):
    scene_payload_key = None
    if state.response.latest_scene_payload is not None:
        scene_payload_key = json.dumps(
            state.response.latest_scene_payload,
            sort_keys=True,
            ensure_ascii=True,
        )

    now_s = time.time()
    payload_changed = (
        scene_payload_key is not None
        and scene_payload_key != state.response.last_scene_payload_key
    )
    cooldown_ready = (
        now_s - state.response.last_spatial_response_time
        >= config.spatial_response_cooldown_s
    )
    should_generate_spatial_response = (
        state.tracking.tracking_ready
        and state.response.latest_scene_payload is not None
        and not state.response.pending_spatial_response
        and payload_changed
        and cooldown_ready
    )
    if not should_generate_spatial_response:
        return False

    state.response.pending_spatial_response = True
    state.response.last_scene_payload_key = scene_payload_key
    state.response.last_spatial_response_time = now_s
    print(
        "sending spatial response request:",
        {
            "ego_relation": state.response.latest_scene_payload.get("ego_relation"),
            "relations": state.response.latest_scene_payload.get("relations"),
            "target_depth_m": state.response.latest_scene_payload.get(
                "target_depth_m"
            ),
        },
    )
    asyncio.create_task(
        spatial_response_handler(
            state.response.latest_scene_payload,
            state.spatial_response_queue,
            llm,
        )
    )
    return True
