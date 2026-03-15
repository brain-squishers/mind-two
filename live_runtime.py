import ast
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


if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def _normalize_phrase(value):
    if not isinstance(value, str):
        return ""
    return value.strip().strip(".").strip()


def _normalize_phrase_list(values, max_items=None):
    if not isinstance(values, list):
        return []

    normalized = []
    seen = set()
    for value in values:
        phrase = _normalize_phrase(value)
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(phrase)
        if max_items is not None and len(normalized) >= max_items:
            break
    return normalized


def parse_extraction_payload(raw_text):
    payload = ast.literal_eval(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("LLM extraction output must be a dictionary")

    targets = _normalize_phrase_list(payload.get("targets", []), max_items=3)
    anchors = _normalize_phrase_list(payload.get("anchors", []))
    support_surfaces = _normalize_phrase_list(payload.get("support_surfaces", []))

    if not targets:
        legacy_target = _normalize_phrase(payload.get("target", ""))
        if legacy_target:
            targets = [legacy_target]

    if not targets:
        legacy_query = _normalize_phrase(payload.get("query", ""))
        if legacy_query:
            targets = [legacy_query]

    if not targets:
        raise ValueError("LLM extraction output is missing valid targets")

    return {
        "targets": targets,
        "anchors": anchors,
        "support_surfaces": support_surfaces,
    }


async def extract(query, model):
    with open("llm/openie.txt", "r") as file:
        ie_prompt = file.read()
    raw_text = await model.generate(ie_prompt.format_map({"query": query}))
    return parse_extraction_payload(raw_text)


async def extract_handler(query, queue, model):
    extraction = await extract(query, model)
    queue.append(extraction)


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

    def submit(
        self,
        frame,
        text,
        request_id,
        job_type="target",
        box_threshold=None,
        text_threshold=None,
    ):
        with self.lock:
            if self.busy:
                return False
            self.pending_job = {
                "frame": frame.copy(),
                "text": text,
                "request_id": request_id,
                "job_type": job_type,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
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
                job_box_threshold = job["box_threshold"]
                if job_box_threshold is None:
                    job_box_threshold = self.box_threshold

                job_text_threshold = job["text_threshold"]
                if job_text_threshold is None:
                    job_text_threshold = self.text_threshold

                result = run_grounding(
                    self.grounding_processor,
                    self.grounding_model,
                    job["frame"],
                    job["text"],
                    box_threshold=job_box_threshold,
                    text_threshold=job_text_threshold,
                )
            except Exception as e:
                with self.lock:
                    self.latest_result = {
                        "result": None,
                        "frame": job["frame"],
                        "text": job["text"],
                        "request_id": job["request_id"],
                        "job_type": job["job_type"],
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
                    "job_type": job["job_type"],
                    "error": None,
                }
                self.busy = False

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


class DepthWorker:
    def __init__(self, depth_model, input_size=518):
        self.depth_model = depth_model
        self.input_size = input_size
        self.lock = threading.Lock()
        self.pending_job = None
        self.latest_result = None
        self.running_job = False
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame, request_id):
        with self.lock:
            self.pending_job = {
                "frame": frame.copy(),
                "request_id": request_id,
            }
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
                if not self.running_job and self.pending_job is not None:
                    job = self.pending_job
                    self.pending_job = None
                    self.running_job = True

            if job is None:
                time.sleep(0.005)
                continue

            try:
                depth_map = infer_metric_depth(
                    self.depth_model,
                    job["frame"],
                    self.input_size,
                )
                result = {
                    "depth_map": depth_map,
                    "request_id": job["request_id"],
                    "error": None,
                }
            except Exception as e:
                result = {
                    "depth_map": None,
                    "request_id": job["request_id"],
                    "error": str(e),
                }

            with self.lock:
                self.latest_result = result
                self.running_job = False

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
