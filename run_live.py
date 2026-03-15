import argparse
import ast
import asyncio
import collections
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from depth_anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from llm.gpt4o_modeling import GPT4o
from llm.qwen2_modeling import Qwen2
from sam2.build_sam import build_sam2_camera_predictor
from utils import add_text_with_background


if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


async def extract(query, model):
    with open("llm/openie.txt", "r") as file:
        ie_prompt = file.read()
    text = await model.generate(ie_prompt.format_map({"query": query}))
    text = ast.literal_eval(text)["query"]
    return text


async def extract_handler(query, queue, model):
    text = await extract(query, model)
    queue.append(text)


def load_model(model):
    model_id = "gdino_checkpoints/grounding-dino-tiny"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )

    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)

    if "gpt" in model.lower():
        llm = GPT4o(model)
    elif "qwen" in model.lower():
        llm = Qwen2(f"llm_checkpoints/{model}", device=device)
    else:
        raise NotImplementedError("INVALID MODEL NAME")

    return grounding_processor, grounding_model, predictor, llm


def get_depth_checkpoint_path(depth_dataset, depth_encoder):
    return (
        "depth_anything_checkpoints/"
        f"depth_anything_v2_metric_{depth_dataset}_{depth_encoder}.pth"
    )


def load_depth_model(
    depth_encoder,
    depth_dataset,
    depth_max_depth,
    depth_checkpoint=None,
):
    checkpoint_path = depth_checkpoint or get_depth_checkpoint_path(
        depth_dataset, depth_encoder
    )
    depth_model = DepthAnythingV2(
        **{
            **DEPTH_MODEL_CONFIGS[depth_encoder],
            "max_depth": depth_max_depth,
        }
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    depth_model.load_state_dict(state_dict)
    depth_model = depth_model.to(device).eval()
    return depth_model, checkpoint_path


def infer_metric_depth(depth_model, frame, input_size):
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                depth = depth_model.infer_image(frame, input_size)
        else:
            depth = depth_model.infer_image(frame, input_size)
    return depth


def compute_mask_depth_stats(
    depth_map,
    mask,
    min_mask_pixels=300,
    erode_kernel=None,
    max_depth=None,
):
    mask_uint8 = mask.astype(np.uint8)
    if erode_kernel is not None:
        mask_uint8 = cv2.erode(mask_uint8, erode_kernel, iterations=1)

    valid_mask_pixels = int(np.count_nonzero(mask_uint8))
    if valid_mask_pixels < min_mask_pixels:
        return None

    depth_values = depth_map[mask_uint8 > 0]
    depth_values = depth_values[np.isfinite(depth_values)]
    depth_values = depth_values[depth_values > 0]

    if max_depth is not None:
        depth_values = depth_values[depth_values < max_depth]

    if depth_values.size == 0:
        return None

    return {
        "median_m": float(np.median(depth_values)),
        "p25_m": float(np.percentile(depth_values, 25)),
        "p75_m": float(np.percentile(depth_values, 75)),
        "count": int(depth_values.size),
    }


class LatestFrameCapture:
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_index}")

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class FpsTracker:
    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def tick(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now
        return self.fps


def run_grounding(
    grounding_processor,
    grounding_model,
    frame,
    text,
    box_threshold=0.35,
    text_threshold=0.25,
):
    inputs = grounding_processor(
        images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        text=text,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = grounding_model(**inputs)
        else:
            outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[frame.shape[:2]],
    )
    return results[0]


class GroundingWorker:
    def __init__(
        self,
        grounding_processor,
        grounding_model,
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        self.grounding_processor = grounding_processor
        self.grounding_model = grounding_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.lock = threading.Lock()
        self.pending_job = None
        self.latest_result = None
        self.busy = False
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame, text, request_id):
        with self.lock:
            if self.busy:
                return False
            self.pending_job = {
                "frame": frame.copy(),
                "text": text,
                "request_id": request_id,
            }
            self.busy = True
            return True

    def poll_result(self):
        with self.lock:
            result = self.latest_result
            self.latest_result = None
            return result

    def _run(self):
        while not self.stopped:
            job = None
            with self.lock:
                if self.pending_job is not None:
                    job = self.pending_job
                    self.pending_job = None

            if job is None:
                time.sleep(0.005)
                continue

            try:
                result = run_grounding(
                    self.grounding_processor,
                    self.grounding_model,
                    job["frame"],
                    job["text"],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
            except Exception as e:
                with self.lock:
                    self.latest_result = {
                        "result": None,
                        "frame": job["frame"],
                        "text": job["text"],
                        "request_id": job["request_id"],
                        "error": str(e),
                    }
                    self.busy = False
                continue

            with self.lock:
                self.latest_result = {
                    "result": result,
                    "frame": job["frame"],
                    "text": job["text"],
                    "request_id": job["request_id"],
                    "error": None,
                }
                self.busy = False

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


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
    depth_every=5,
    depth_input_size=518,
    depth_min_mask_pixels=300,
    depth_mask_erode_kernel=5,
):
    grounding_processor, grounding_model, predictor, llm = load_model(model)
    depth_model = None
    if enable_depth:
        depth_model, depth_checkpoint_path = load_depth_model(
            depth_encoder=depth_encoder,
            depth_dataset=depth_dataset,
            depth_max_depth=depth_max_depth,
            depth_checkpoint=depth_checkpoint,
        )
        print(f"depth checkpoint {depth_checkpoint_path}")

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
    tracking_ready = False
    last_mask_rgb = None
    low_mask_count = 0
    frame_count = 0
    processed_frames = 0
    tracking_steps = 0
    fps_tracker = FpsTracker()
    pending_detection_request = None
    force_redetect = True
    latest_depth_map = None
    latest_depth_frame_idx = -1
    latest_depth_stats = {}
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
                        for i, box in enumerate(boxes):
                            predictor.add_new_points(
                                frame_idx=0,
                                obj_id=i + 1,
                                box=box,
                            )
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
                tracking_ready = False
                last_mask_rgb = None
                low_mask_count = 0
                tracking_steps = 0
                pending_detection_request = None
                force_redetect = True
                latest_depth_map = None
                latest_depth_frame_idx = -1
                latest_depth_stats = {}

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
                    should_run_depth = enable_depth and (
                        latest_depth_map is None
                        or tracking_steps == 0
                        or (
                            depth_every > 0 and processed_frames % depth_every == 0
                        )
                    )
                    if should_run_depth:
                        latest_depth_map = infer_metric_depth(
                            depth_model,
                            frame,
                            depth_input_size,
                        )
                        latest_depth_frame_idx = processed_frames

                    tracking_steps += 1
                    should_track = tracking_steps == 1 or (
                        track_every > 0 and tracking_steps % track_every == 0
                    )
                    if should_track:
                        out_obj_ids, out_mask_logits = predictor.track(frame)
                        width, height = frame.shape[:2][::-1]
                        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                        current_depth_stats = {}
                        for i in range(len(out_obj_ids)):
                            binary_mask = (
                                (out_mask_logits[i] > 0.0)
                                .squeeze(0)
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            out_mask = binary_mask[:, :, None] * 255
                            all_mask = cv2.bitwise_or(all_mask, out_mask)
                            if enable_depth and latest_depth_map is not None:
                                stats = compute_mask_depth_stats(
                                    latest_depth_map,
                                    binary_mask,
                                    min_mask_pixels=depth_min_mask_pixels,
                                    erode_kernel=depth_kernel,
                                    max_depth=depth_max_depth,
                                )
                                if stats is not None:
                                    current_depth_stats[int(out_obj_ids[i])] = stats

                        if len(out_obj_ids) != 0:
                            latest_depth_stats = current_depth_stats
                            last_mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                            mask_pixels = int(np.count_nonzero(all_mask))
                            mask_ratio = mask_pixels / float(width * height)
                            print(
                                f"track mask_pixels: {mask_pixels} mask_ratio: {mask_ratio:.4f}"
                            )
                            print("cache updated")
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
                                print("tracking lost; returning to detection")
                        else:
                            last_mask_rgb = None
                            latest_depth_stats = {}
                            low_mask_count += 1
                            print("track produced no object ids; cache cleared")
                            if low_mask_count >= lost_patience:
                                tracking_ready = False
                                low_mask_count = 0
                                pending_detection_request = None
                                force_redetect = True
                                tracking_steps = 0
                                latest_depth_stats = {}
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
                    f"DEPTH {latest_depth_frame_idx}"
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

            cv2.imshow("run_live", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            await asyncio.sleep(0)

    finally:
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
    parser.add_argument("--depth-every", type=int, default=5)
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--depth-min-mask-pixels", type=int, default=300)
    parser.add_argument("--depth-mask-erode-kernel", type=int, default=5)
    # parser.add_argument("--query", type=str, default="I am trying to find my glass")
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
        )
    )
