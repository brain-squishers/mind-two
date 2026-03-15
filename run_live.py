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


def run_grounding(grounding_processor, grounding_model, frame, text):
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
        threshold=0.6,
        text_threshold=0.6,
        target_sizes=[frame.shape[:2]],
    )
    return results[0]["boxes"]


class GroundingWorker:
    def __init__(self, grounding_processor, grounding_model):
        self.grounding_processor = grounding_processor
        self.grounding_model = grounding_model
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
                boxes = run_grounding(
                    self.grounding_processor,
                    self.grounding_model,
                    job["frame"],
                    job["text"],
                )
            except Exception as e:
                with self.lock:
                    self.latest_result = {
                        "boxes": None,
                        "frame": job["frame"],
                        "text": job["text"],
                        "request_id": job["request_id"],
                        "error": str(e),
                    }
                    self.busy = False
                continue

            with self.lock:
                self.latest_result = {
                    "boxes": boxes,
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
    query="I am trying to find my glass",
):
    grounding_processor, grounding_model, predictor, llm = load_model(model)
    capture = LatestFrameCapture(camera_index).start()
    grounding_worker = GroundingWorker(grounding_processor, grounding_model).start()

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
                    print(f"redetect boxes: {detection_result['boxes'].shape[0]}")
                    if detection_result["boxes"].shape[0] != 0:
                        predictor.load_first_frame(detection_result["frame"])
                        predictor.add_new_points(
                            frame_idx=0,
                            obj_id=2,
                            box=detection_result["boxes"],
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

            should_process = frame_count == 1 or frame_count % skip_frames == 0

            if text and should_process:
                processed_frames += 1
                if not tracking_ready:
                    should_redetect = force_redetect or processed_frames == 1 or (
                        init_redetect_every > 0
                        and processed_frames % init_redetect_every == 0
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
                    tracking_steps += 1
                    should_track = tracking_steps == 1 or (
                        track_every > 0 and tracking_steps % track_every == 0
                    )
                    if should_track:
                        out_obj_ids, out_mask_logits = predictor.track(frame)
                        width, height = frame.shape[:2][::-1]
                        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                        for i in range(len(out_obj_ids)):
                            out_mask = (out_mask_logits[i] > 0.0).permute(
                                1, 2, 0
                            ).cpu().numpy().astype(np.uint8) * 255
                            all_mask = cv2.bitwise_or(all_mask, out_mask)

                        if len(out_obj_ids) != 0:
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
                                print("tracking lost; returning to detection")
                        else:
                            last_mask_rgb = None
                            low_mask_count += 1
                            print("track produced no object ids; cache cleared")
                            if low_mask_count >= lost_patience:
                                tracking_ready = False
                                low_mask_count = 0
                                pending_detection_request = None
                                force_redetect = True
                                tracking_steps = 0
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
    # parser.add_argument("--query", type=str, default="I am trying to find my glass")
    parser.add_argument("--query", type=str, default="Where is my can of juice?")
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
            query=args.query,
        )
    )
