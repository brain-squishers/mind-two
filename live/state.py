import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class QueryState:
    current_query: str
    current_query_summary: str
    target_queries: list[str] = field(default_factory=list)
    target_query_index: int = 0
    anchor_queries: list[str] = field(default_factory=list)
    support_surface_queries: list[str] = field(default_factory=list)
    active_target_query: str = ""
    active_anchor_query: str = ""
    active_support_query: str = ""
    current_label: str = "object"


@dataclass(slots=True)
class TrackingState:
    tracking_ready: bool = False
    last_mask_rgb: np.ndarray | None = None
    low_mask_count: int = 0
    tracking_steps: int = 0
    obj_display_names: dict[int, str] = field(default_factory=dict)
    latest_label_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    latest_target_boxes: list[np.ndarray] = field(default_factory=list)
    pending_detection_request: int | None = None
    force_redetect: bool = True


@dataclass(slots=True)
class DepthState:
    latest_depth_map: np.ndarray | None = None
    latest_depth_frame_idx: int = -1
    latest_depth_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    pending_depth_request: int | None = None
    latest_mask_ratio: float = 0.0
    stable_mask_count: int = 0


@dataclass(slots=True)
class ContextState:
    latest_anchor_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_support_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_hand_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_anchor_frame_idx: int = -1
    latest_support_frame_idx: int = -1
    latest_hand_frame_idx: int = -1
    pending_anchor_request: int | None = None
    pending_support_request: int | None = None
    pending_hand_request: int | None = None


@dataclass(slots=True)
class ResponseState:
    latest_relations: list[dict[str, Any]] = field(default_factory=list)
    latest_spatial_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_scene_payload: dict[str, Any] | None = None
    latest_spatial_response: str = ""
    pending_spatial_response: bool = False
    last_scene_payload_key: str | None = None
    last_spatial_response_time: float = 0.0


@dataclass(slots=True)
class MemoryState:
    latest_memory_entry: Any | None = None
    latest_memory_response: str = ""
    last_memory_write_time: float = 0.0


@dataclass(slots=True)
class RuntimeState:
    query: QueryState
    tracking: TrackingState = field(default_factory=TrackingState)
    depth: DepthState = field(default_factory=DepthState)
    context: ContextState = field(default_factory=ContextState)
    response: ResponseState = field(default_factory=ResponseState)
    memory: MemoryState = field(default_factory=MemoryState)
    frame_count: int = 0
    processed_frames: int = 0
    query_queue: collections.deque[str] = field(default_factory=collections.deque)
    response_queue: collections.deque[dict[str, Any]] = field(
        default_factory=collections.deque
    )
    spatial_response_queue: collections.deque[str] = field(
        default_factory=collections.deque
    )


def build_initial_runtime_state(initial_query: str) -> RuntimeState:
    return RuntimeState(
        query=QueryState(
            current_query=initial_query,
            current_query_summary=initial_query,
        ),
        query_queue=collections.deque(),
        response_queue=collections.deque(),
        spatial_response_queue=collections.deque(),
    )
