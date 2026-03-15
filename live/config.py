from dataclasses import dataclass


DEFAULT_HAND_QUERY = "hand"


@dataclass(slots=True)
class LiveConfig:
    model: str = "gpt-4o-2024-05-13"
    camera_index: int = 0
    skip_frames: int = 2
    track_every: int = 3
    init_redetect_every: int = 10
    redetect_every: int = 30
    lost_mask_ratio: float = 0.001
    lost_patience: int = 5
    max_objects: int = 3
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    target_box_threshold: float = 0.36
    target_text_threshold: float = 0.30
    query: str = "I am trying to find my glass"
    enable_depth: bool = True
    depth_encoder: str = "vits"
    depth_dataset: str = "hypersim"
    depth_max_depth: float = 20.0
    depth_checkpoint: str | None = None
    depth_every: int = 10
    depth_input_size: int = 336
    depth_min_mask_pixels: int = 300
    depth_mask_erode_kernel: int = 5
    depth_stable_mask_ratio: float = 0.01
    depth_stable_patience: int = 2
    anchor_redetect_every: int = 45
    support_redetect_every: int = 75
    anchor_cache_ttl: int = 60
    support_cache_ttl: int = 90
    enable_hand_detection: bool = False
    hand_query: str = DEFAULT_HAND_QUERY
    hand_redetect_every: int = 10
    hand_cache_ttl: int = 25
    hand_box_threshold: float = 0.25
    hand_text_threshold: float = 0.20
    spatial_response_cooldown_s: float = 2.5
    scene_memory_retention_seconds: float = 60.0
    scene_memory_merge_time_window_s: float = 2.0
    scene_memory_merge_distance_m: float = 0.5
    scene_memory_max_entries: int = 200
    memory_write_cooldown_s: float = 1.5

