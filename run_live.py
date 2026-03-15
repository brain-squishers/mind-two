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
from utils import add_text_with_background


print("device", device)


def format_query_label(text):
    label = re.sub(r"[\s\.,;:!?]+$", "", text.strip())
    return label or "object"


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
    text = ""
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

    frame_count = 0
    processed_frames = 0
    pending_detection_request = None
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
                    print(f"redetect error: {detection_result['error']}")
                    pending_detection_request = None
                    force_redetect = True
                    detection_result = None
                if detection_result is not None and detection_result["text"] == text:
                    result = detection_result["result"]
                    boxes = result["boxes"]
                    scores = result["scores"]
                    print(f"redetect boxes: {boxes.shape[0]}")
                    if boxes.shape[0] != 0:
                        keep_count = min(max_objects, boxes.shape[0])
                        if keep_count < boxes.shape[0]:
                            top_indices = torch.argsort(scores, descending=True)[
                                :keep_count
                            ]
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
                            obj_display_names[obj_id] = f"{current_label} #{obj_id}"
                        tracking_ready = True
                        last_mask_rgb = None
                        low_mask_count = 0
                        tracking_steps = 0
                        force_redetect = False
                pending_detection_request = None

            if query_queue:
                pending_query = query_queue.popleft()
                asyncio.create_task(extract_handler(pending_query, response_queue, llm))

            if response_queue:
                text = response_queue.popleft()
                current_label = format_query_label(text)
                tracking_ready = False
                last_mask_rgb = None
                low_mask_count = 0
                tracking_steps = 0
                pending_detection_request = None
                force_redetect = True
                latest_depth_map = None
                latest_depth_frame_idx = -1
                latest_depth_stats = {}
                pending_depth_request = None
                latest_mask_ratio = 0.0
                stable_mask_count = 0
                obj_display_names = {}
                latest_label_positions = {}

            should_process = frame_count == 1 or frame_count % skip_frames == 0

            if text and should_process:
                processed_frames += 1
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
                        submitted = grounding_worker.submit(frame, text, request_id)
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
                                print("tracking lost; returning to detection")
                        else:
                            last_mask_rgb = None
                            latest_depth_stats = {}
                            latest_mask_ratio = 0.0
                            stable_mask_count = 0
                            latest_label_positions = {}
                            low_mask_count += 1
                            print("track produced no object ids; cache cleared")
                            if low_mask_count >= lost_patience:
                                tracking_ready = False
                                low_mask_count = 0
                                pending_detection_request = None
                                force_redetect = True
                                tracking_steps = 0
                                latest_depth_stats = {}
                                latest_mask_ratio = 0.0
                                stable_mask_count = 0
                                latest_label_positions = {}
                                print("tracking lost; returning to detection")

            if last_mask_rgb is not None:
                display_frame = cv2.addWeighted(display_frame, 1, last_mask_rgb, 0.5, 0)

            if current_query:
                display_frame = add_text_with_background(display_frame, current_query)

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
    parser.add_argument("--text-threshold", type=float, default=0.6)
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
    parser.add_argument("--query", type=str, default="I am trying to find phones")
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
        )
    )
