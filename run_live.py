import argparse
import asyncio

import cv2

from live.bootstrap import build_runtime_dependencies, release_runtime_dependencies
from live.config import DEFAULT_HAND_QUERY, LiveConfig
from live.context_pipeline import (
    apply_context_detection_result,
    expire_context_cache,
    schedule_context_detections,
)
from live.depth_pipeline import build_depth_kernel, maybe_schedule_depth, poll_depth_result
from live.memory_pipeline import maybe_write_memory, refresh_memory_response_if_needed
from live.overlay_renderer import render_display_frame
from live.query_pipeline import (
    apply_extraction_result,
    poll_query_input,
    schedule_query_extraction,
)
from live.scene_pipeline import (
    maybe_schedule_spatial_response,
    poll_spatial_response,
    update_scene_state,
)
from live.state import build_initial_runtime_state
from live.tracking_pipeline import (
    handle_detection_error,
    handle_target_detection_result,
    maybe_schedule_target_detection,
    run_tracking_step,
)
from live_runtime import (
    FpsTracker,
    compute_mask_depth_stats,
    device,
    extract_handler,
    spatial_response_handler,
)
from scene_memory import SceneMemory


print("device", device)


def get_ego_relation(relations):
    for relation in relations:
        if relation["type"] in {"on_your_left", "on_your_right", "straight_ahead"}:
            return relation["type"]
    return None


async def main(config: LiveConfig):
    deps = build_runtime_dependencies(config)
    state = build_initial_runtime_state(config.query)
    fps_tracker = FpsTracker()
    depth_kernel = build_depth_kernel(config)
    scene_memory = SceneMemory(
        retention_seconds=config.scene_memory_retention_seconds,
        merge_time_window_s=config.scene_memory_merge_time_window_s,
        merge_distance_m=config.scene_memory_merge_distance_m,
        max_entries=config.scene_memory_max_entries,
    )

    try:
        while True:
            poll_query_input(deps.query_input, state)

            frame = deps.frame_source.read_latest()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            state.frame_count += 1
            poll_depth_result(deps.depth_worker, state.depth)

            detection_result = deps.grounding_worker.poll_result()
            detection_result = handle_detection_error(detection_result, state)
            if not handle_target_detection_result(
                detection_result,
                deps.predictor,
                state,
                config,
            ):
                apply_context_detection_result(detection_result, state, config)

            schedule_query_extraction(state, deps.llm, extract_handler)
            poll_spatial_response(state, deps.output_sink)

            extraction = apply_extraction_result(state)
            if extraction is not None:
                print("extraction:", extraction)

            should_process = (
                state.frame_count == 1 or state.frame_count % config.skip_frames == 0
            )
            if state.query.active_target_query and should_process:
                state.processed_frames += 1
                expire_context_cache(state, config)

                target_redetect_submitted = maybe_schedule_target_detection(
                    frame,
                    deps.grounding_worker,
                    state,
                    config,
                )
                if (not target_redetect_submitted) and state.tracking.tracking_ready:
                    maybe_schedule_depth(frame, deps.depth_worker, state, config)
                    run_tracking_step(
                        frame,
                        deps.predictor,
                        compute_mask_depth_stats,
                        depth_kernel,
                        scene_memory,
                        state,
                        config,
                    )

                schedule_context_detections(
                    frame, deps.grounding_worker, state, config
                )
                update_scene_state(state, config, frame.shape)
                maybe_write_memory(scene_memory, state, config, get_ego_relation)
                maybe_schedule_spatial_response(
                    state,
                    config,
                    deps.llm,
                    spatial_response_handler,
                )

            refresh_memory_response_if_needed(state)
            display_frame = render_display_frame(
                frame,
                query_state=state.query,
                tracking_state=state.tracking,
                depth_state=state.depth,
                context_state=state.context,
                response_state=state.response,
                memory_state=state.memory,
                fps=fps_tracker.tick(),
                enable_depth=config.enable_depth,
            )

            cv2.imshow("run_live", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            await asyncio.sleep(0)

    finally:
        release_runtime_dependencies(deps)
        cv2.destroyAllWindows()


def build_arg_parser():
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
    parser.add_argument("--target-box-threshold", type=float, default=0.36)
    parser.add_argument("--target-text-threshold", type=float, default=0.30)
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
    parser.add_argument(
        "--disable-hand-detection",
        action="store_false",
        dest="enable_hand_detection",
        help="Disable hand detections and hand-relative relations.",
    )
    parser.set_defaults(enable_hand_detection=False)
    parser.add_argument("--hand-query", type=str, default=DEFAULT_HAND_QUERY)
    parser.add_argument("--hand-redetect-every", type=int, default=10)
    parser.add_argument("--hand-cache-ttl", type=int, default=25)
    parser.add_argument("--hand-box-threshold", type=float, default=0.25)
    parser.add_argument("--hand-text-threshold", type=float, default=0.20)
    parser.add_argument("--query", type=str, default="I am trying to find phones")
    return parser


def config_from_args(args):
    return LiveConfig(
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
        enable_hand_detection=args.enable_hand_detection,
        hand_query=args.hand_query,
        hand_redetect_every=args.hand_redetect_every,
        hand_cache_ttl=args.hand_cache_ttl,
        hand_box_threshold=args.hand_box_threshold,
        hand_text_threshold=args.hand_text_threshold,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main(config_from_args(args)))
