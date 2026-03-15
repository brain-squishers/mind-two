import asyncio
import re


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


def poll_query_input(query_input, state):
    new_query = query_input.poll_query()
    if not new_query:
        return None

    state.query_queue.append(new_query)
    state.query.current_query = new_query
    state.query.current_query_summary = new_query
    return new_query


def schedule_query_extraction(state, llm, extract_handler):
    if not state.query_queue:
        return False

    pending_query = state.query_queue.popleft()
    asyncio.create_task(extract_handler(pending_query, state.response_queue, llm))
    return True


def apply_extraction_result(state):
    if not state.response_queue:
        return None

    extraction = state.response_queue.popleft()
    query_state = state.query
    tracking_state = state.tracking
    depth_state = state.depth
    context_state = state.context
    response_state = state.response
    memory_state = state.memory

    query_state.target_queries = extraction["targets"]
    query_state.target_query_index = 0
    query_state.anchor_queries = extraction["anchors"]
    query_state.support_surface_queries = extraction["support_surfaces"]
    query_state.active_target_query = get_active_target_query(
        query_state.target_queries,
        query_state.target_query_index,
    )
    query_state.active_anchor_query = build_phrase_query(query_state.anchor_queries)
    query_state.active_support_query = build_phrase_query(
        query_state.support_surface_queries
    )
    query_state.current_label = format_query_label(query_state.active_target_query)
    query_state.current_query_summary = format_query_summary(
        query_state.target_queries,
        query_state.anchor_queries,
        query_state.support_surface_queries,
        active_target_query=query_state.active_target_query,
    )

    tracking_state.tracking_ready = False
    tracking_state.last_mask_rgb = None
    tracking_state.low_mask_count = 0
    tracking_state.tracking_steps = 0
    tracking_state.obj_display_names = {}
    tracking_state.latest_label_positions = {}
    tracking_state.latest_target_boxes = []
    tracking_state.pending_detection_request = None
    tracking_state.force_redetect = True

    depth_state.latest_depth_map = None
    depth_state.latest_depth_frame_idx = -1
    depth_state.latest_depth_stats = {}
    depth_state.pending_depth_request = None
    depth_state.latest_mask_ratio = 0.0
    depth_state.stable_mask_count = 0

    context_state.latest_anchor_detections = []
    context_state.latest_support_detections = []
    context_state.latest_hand_detections = []
    context_state.latest_anchor_frame_idx = -1
    context_state.latest_support_frame_idx = -1
    context_state.latest_hand_frame_idx = -1
    context_state.pending_anchor_request = None
    context_state.pending_support_request = None
    context_state.pending_hand_request = None

    response_state.latest_relations = []
    response_state.latest_spatial_index = {}
    response_state.latest_scene_payload = None
    response_state.latest_spatial_response = ""
    response_state.pending_spatial_response = False
    response_state.last_scene_payload_key = None
    response_state.last_spatial_response_time = 0.0

    memory_state.latest_memory_entry = None
    memory_state.latest_memory_response = ""

    return extraction


def advance_target_query_candidate(query_state):
    if query_state.target_query_index + 1 >= len(query_state.target_queries):
        return False

    query_state.target_query_index += 1
    query_state.active_target_query = get_active_target_query(
        query_state.target_queries,
        query_state.target_query_index,
    )
    query_state.current_label = format_query_label(query_state.active_target_query)
    query_state.current_query_summary = format_query_summary(
        query_state.target_queries,
        query_state.anchor_queries,
        query_state.support_surface_queries,
        active_target_query=query_state.active_target_query,
    )
    return True
