import numpy as np


def build_depth_kernel(config):
    if config.depth_mask_erode_kernel <= 1:
        return None
    return np.ones(
        (config.depth_mask_erode_kernel, config.depth_mask_erode_kernel),
        dtype=np.uint8,
    )


def poll_depth_result(depth_worker, depth_state):
    if depth_worker is None:
        return None

    depth_result = depth_worker.poll_result()
    if depth_result is None:
        return None

    if depth_result.get("error") is not None:
        print(f"depth error: {depth_result['error']}")
    else:
        depth_state.latest_depth_map = depth_result["depth_map"]
        depth_state.latest_depth_frame_idx = depth_result["request_id"]
    depth_state.pending_depth_request = None
    return depth_result


def maybe_schedule_depth(frame, depth_worker, state, config):
    if depth_worker is None:
        return False

    tracking_state = state.tracking
    depth_state = state.depth

    depth_tracking_stable = (
        tracking_state.last_mask_rgb is not None
        and depth_state.latest_mask_ratio >= config.depth_stable_mask_ratio
        and depth_state.stable_mask_count >= config.depth_stable_patience
    )
    should_run_depth = config.enable_depth and (
        depth_state.latest_depth_map is None
        or (depth_tracking_stable and tracking_state.tracking_steps == 0)
        or (
            depth_tracking_stable
            and config.depth_every > 0
            and state.processed_frames % config.depth_every == 0
        )
    )
    if not should_run_depth:
        return False

    request_id = state.processed_frames
    if depth_state.pending_depth_request == request_id:
        return False

    depth_worker.submit(frame, request_id)
    depth_state.pending_depth_request = request_id
    return True
