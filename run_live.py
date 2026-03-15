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
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.6,
        text_threshold=0.6,
        target_sizes=[frame.shape[:2]],
    )
    return results[0]["boxes"]


async def main(
    model="gpt-4o-2024-05-13",
    camera_index=0,
    skip_frames=2,
    init_redetect_every=10,
    redetect_every=30,
    query="I am trying to find my glass",
):
    grounding_processor, grounding_model, predictor, llm = load_model(model)
    capture = LatestFrameCapture(camera_index).start()

    query_queue = collections.deque([query])
    response_queue = collections.deque([])
    current_query = query
    text = ""
    tracking_ready = False
    last_mask_rgb = None
    frame_count = 0
    processed_frames = 0
    fps_tracker = FpsTracker()

    try:
        while True:
            frame = capture.read_latest()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            frame_count += 1
            display_frame = frame.copy()

            if query_queue:
                pending_query = query_queue.popleft()
                asyncio.create_task(extract_handler(pending_query, response_queue, llm))

            if response_queue:
                text = response_queue.popleft()
                tracking_ready = False
                last_mask_rgb = None

            should_process = frame_count == 1 or frame_count % skip_frames == 0

            if text and should_process:
                processed_frames += 1
                if not tracking_ready:
                    should_redetect = processed_frames == 1 or (
                        init_redetect_every > 0
                        and processed_frames % init_redetect_every == 0
                    )
                else:
                    should_redetect = (
                        redetect_every > 0 and processed_frames % redetect_every == 0
                    )

                if should_redetect:
                    boxes = run_grounding(
                        grounding_processor, grounding_model, frame, text
                    )
                    print(f"redetect boxes: {boxes.shape[0]}")
                    if boxes.shape[0] != 0:
                        predictor.load_first_frame(frame)
                        predictor.add_new_points(frame_idx=0, obj_id=2, box=boxes)
                        tracking_ready = True
                        last_mask_rgb = None
                elif tracking_ready:
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
                    else:
                        last_mask_rgb = None
                        print("track produced no object ids; cache cleared")

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

            cv2.imshow("run_live", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            await asyncio.sleep(0)

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-latency live webcam runner.")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--skip-frames", type=int, default=2)
    parser.add_argument("--init-redetect-every", type=int, default=10)
    parser.add_argument("--redetect-every", type=int, default=30)
    # parser.add_argument("--query", type=str, default="I am trying to find my glass")
    parser.add_argument("--query", type=str, default="Where is my can of juice?")
    args = parser.parse_args()

    asyncio.run(
        main(
            model=args.model,
            camera_index=args.camera_index,
            skip_frames=args.skip_frames,
            init_redetect_every=args.init_redetect_every,
            redetect_every=args.redetect_every,
            query=args.query,
        )
    )
