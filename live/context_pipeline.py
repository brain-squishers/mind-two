import re

from spatial_reasoning import compute_iou_xyxy


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


def apply_context_detection_result(detection_result, state, config):
    if detection_result is None:
        return False

    query_state = state.query
    tracking_state = state.tracking
    context_state = state.context

    job_type = detection_result["job_type"]
    result = detection_result["result"]

    if job_type == "anchor":
        if detection_result["text"] == query_state.active_anchor_query:
            context_state.latest_anchor_detections = suppress_target_overlaps(
                extract_labeled_boxes(result, query_state.anchor_queries),
                tracking_state.latest_target_boxes,
                query_state.target_queries,
            )
            context_state.latest_anchor_frame_idx = state.processed_frames
        context_state.pending_anchor_request = None
        return True

    if job_type == "support":
        if detection_result["text"] == query_state.active_support_query:
            context_state.latest_support_detections = suppress_target_overlaps(
                extract_labeled_boxes(result, query_state.support_surface_queries),
                tracking_state.latest_target_boxes,
                query_state.target_queries,
            )
            context_state.latest_support_frame_idx = state.processed_frames
        context_state.pending_support_request = None
        return True

    if job_type == "hand":
        if detection_result["text"] == config.hand_query:
            print(f"hand boxes: {result['boxes'].shape[0]}")
            context_state.latest_hand_detections = extract_labeled_boxes(
                result,
                [config.hand_query],
            )
            context_state.latest_hand_frame_idx = state.processed_frames
        context_state.pending_hand_request = None
        return True

    return False


def expire_context_cache(state, config):
    context_state = state.context

    if (
        context_state.latest_anchor_frame_idx >= 0
        and config.anchor_cache_ttl > 0
        and state.processed_frames - context_state.latest_anchor_frame_idx
        > config.anchor_cache_ttl
    ):
        context_state.latest_anchor_detections = []
        context_state.latest_anchor_frame_idx = -1

    if (
        context_state.latest_support_frame_idx >= 0
        and config.support_cache_ttl > 0
        and state.processed_frames - context_state.latest_support_frame_idx
        > config.support_cache_ttl
    ):
        context_state.latest_support_detections = []
        context_state.latest_support_frame_idx = -1

    if (
        context_state.latest_hand_frame_idx >= 0
        and config.hand_cache_ttl > 0
        and state.processed_frames - context_state.latest_hand_frame_idx
        > config.hand_cache_ttl
    ):
        context_state.latest_hand_detections = []
        context_state.latest_hand_frame_idx = -1


def schedule_context_detections(frame, grounding_worker, state, config):
    query_state = state.query
    tracking_state = state.tracking
    context_state = state.context

    can_schedule_context = (
        tracking_state.pending_detection_request is None
        and query_state.active_target_query
        and tracking_state.tracking_ready
    )

    if (
        config.enable_hand_detection
        and can_schedule_context
        and config.hand_query
        and config.hand_redetect_every > 0
        and state.processed_frames % config.hand_redetect_every == 0
    ):
        request_id = state.processed_frames
        if context_state.pending_hand_request != request_id:
            submitted = grounding_worker.submit(
                frame,
                config.hand_query,
                request_id,
                job_type="hand",
                box_threshold=config.hand_box_threshold,
                text_threshold=config.hand_text_threshold,
            )
            if submitted:
                context_state.pending_hand_request = request_id

    if (
        can_schedule_context
        and query_state.active_anchor_query
        and config.anchor_redetect_every > 0
        and state.processed_frames % config.anchor_redetect_every == 0
    ):
        request_id = state.processed_frames
        if context_state.pending_anchor_request != request_id:
            submitted = grounding_worker.submit(
                frame,
                query_state.active_anchor_query,
                request_id,
                job_type="anchor",
            )
            if submitted:
                context_state.pending_anchor_request = request_id

    if (
        can_schedule_context
        and query_state.active_support_query
        and config.support_redetect_every > 0
        and state.processed_frames % config.support_redetect_every == 0
        and context_state.pending_anchor_request is None
    ):
        request_id = state.processed_frames
        if context_state.pending_support_request != request_id:
            submitted = grounding_worker.submit(
                frame,
                query_state.active_support_query,
                request_id,
                job_type="support",
            )
            if submitted:
                context_state.pending_support_request = request_id
