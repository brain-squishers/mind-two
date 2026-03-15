import time

from scene_memory import format_last_seen_message
from spatial_reasoning import select_top_relations


def apply_memory_fallback(scene_memory, state, max_age_s=30.0):
    query_state = state.query
    memory_state = state.memory
    memory_state.latest_memory_entry = scene_memory.find_best_recent_by_labels(
        query_state.target_queries,
        now_s=time.time(),
        max_age_s=max_age_s,
    )
    memory_state.latest_memory_response = format_last_seen_message(
        memory_state.latest_memory_entry,
        now_s=time.time(),
    )
    if memory_state.latest_memory_response:
        print("memory fallback:", memory_state.latest_memory_response)


def refresh_memory_response_if_needed(state):
    if (not state.tracking.tracking_ready) and state.memory.latest_memory_entry is not None:
        state.memory.latest_memory_response = format_last_seen_message(
            state.memory.latest_memory_entry,
            now_s=time.time(),
        )


def maybe_write_memory(scene_memory, state, config, get_ego_relation):
    target_entry = state.response.latest_spatial_index.get("target_1")
    if target_entry is None:
        return None

    now_s = time.time()
    should_write_memory = (
        state.tracking.tracking_ready
        and target_entry.get("depth_std_m") is not None
        and state.depth.latest_mask_ratio >= config.depth_stable_mask_ratio
        and state.depth.stable_mask_count >= config.depth_stable_patience
        and state.tracking.low_mask_count == 0
        and now_s - state.memory.last_memory_write_time >= config.memory_write_cooldown_s
    )
    if not should_write_memory:
        return None

    top_relations = select_top_relations(state.response.latest_relations)
    memory_entry = scene_memory.add_observation(
        label=state.query.current_label,
        source="target",
        position_cam_3d_m=target_entry["position_3d_m"],
        depth_std_m=target_entry["depth_std_m"],
        ego_relation=get_ego_relation(top_relations),
        relations=top_relations,
        confidence=target_entry.get("confidence"),
        image_center_px=target_entry.get("image_center_px"),
        bbox_xyxy=tuple(state.tracking.latest_target_boxes[0])
        if state.tracking.latest_target_boxes
        else None,
        metadata={
            "target_query": state.query.active_target_query,
        },
        timestamp_s=now_s,
    )
    state.memory.last_memory_write_time = now_s
    print(
        "memory write:",
        {
            "id": memory_entry.memory_id,
            "label": memory_entry.label,
            "ego_relation": memory_entry.ego_relation,
            "position_3d_m": memory_entry.position_cam_3d_m,
        },
    )
    return memory_entry
