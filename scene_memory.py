from dataclasses import dataclass, field
import math
import time
from typing import Any


@dataclass
class MemoryEntry:
    memory_id: str
    label: str
    timestamp_s: float
    source: str
    position_cam_3d_m: tuple[float, float, float] | None
    depth_std_m: float | None
    ego_relation: str | None
    relations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    image_center_px: tuple[float, float] | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _euclidean_distance_3d(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    if a is None or b is None:
        return math.inf
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def _score_memory_entry(
    entry: MemoryEntry,
    now_s: float,
    max_age_s: float,
) -> float:
    age_s = max(0.0, now_s - entry.timestamp_s)
    recency_score = max(0.0, 1.0 - age_s / max(1e-6, max_age_s))

    depth_uncertainty_score = 0.5
    if entry.depth_std_m is not None:
        depth_uncertainty_score = max(0.0, 1.0 - entry.depth_std_m / 0.5)

    ego_score = 1.0 if entry.ego_relation else 0.0
    relation_score = min(1.0, len(entry.relations) / 3.0)

    confidence_score = 0.5
    if entry.confidence is not None:
        confidence_score = max(0.0, min(1.0, float(entry.confidence)))

    return (
        0.40 * recency_score
        + 0.25 * depth_uncertainty_score
        + 0.15 * ego_score
        + 0.10 * relation_score
        + 0.10 * confidence_score
    )


class SceneMemory:
    def __init__(
        self,
        retention_seconds: float = 60.0,
        merge_time_window_s: float = 2.0,
        merge_distance_m: float = 0.5,
        max_entries: int = 200,
    ) -> None:
        self.retention_seconds = retention_seconds
        self.merge_time_window_s = merge_time_window_s
        self.merge_distance_m = merge_distance_m
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []
        self._counter = 0

    def _make_memory_id(self) -> str:
        self._counter += 1
        return f"mem_{self._counter}"

    def _prune(self, now_s: float) -> None:
        self.entries = [
            entry
            for entry in self.entries
            if now_s - entry.timestamp_s <= self.retention_seconds
        ]
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def add_observation(
        self,
        label: str,
        source: str,
        position_cam_3d_m: tuple[float, float, float] | None,
        depth_std_m: float | None,
        ego_relation: str | None,
        relations: list[dict[str, Any]],
        confidence: float | None = None,
        image_center_px: tuple[float, float] | None = None,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp_s: float | None = None,
    ) -> MemoryEntry:
        if timestamp_s is None:
            timestamp_s = time.time()
        if metadata is None:
            metadata = {}

        self._prune(timestamp_s)

        candidate = None
        for entry in reversed(self.entries):
            if entry.label != label:
                continue
            if timestamp_s - entry.timestamp_s > self.merge_time_window_s:
                continue
            if (
                _euclidean_distance_3d(
                    entry.position_cam_3d_m,
                    position_cam_3d_m,
                )
                > self.merge_distance_m
            ):
                continue
            candidate = entry
            break

        if candidate is not None:
            candidate.timestamp_s = timestamp_s
            candidate.position_cam_3d_m = position_cam_3d_m
            candidate.depth_std_m = depth_std_m
            candidate.ego_relation = ego_relation
            candidate.relations = relations
            candidate.confidence = confidence
            candidate.image_center_px = image_center_px
            candidate.bbox_xyxy = bbox_xyxy
            candidate.metadata = metadata
            return candidate

        entry = MemoryEntry(
            memory_id=self._make_memory_id(),
            label=label,
            timestamp_s=timestamp_s,
            source=source,
            position_cam_3d_m=position_cam_3d_m,
            depth_std_m=depth_std_m,
            ego_relation=ego_relation,
            relations=relations,
            confidence=confidence,
            image_center_px=image_center_px,
            bbox_xyxy=bbox_xyxy,
            metadata=metadata,
        )
        self.entries.append(entry)
        return entry

    def find_recent_by_label(
        self,
        label: str,
        now_s: float | None = None,
        max_age_s: float | None = None,
    ) -> MemoryEntry | None:
        if now_s is None:
            now_s = time.time()
        if max_age_s is None:
            max_age_s = self.retention_seconds

        self._prune(now_s)

        for entry in reversed(self.entries):
            if entry.label != label:
                continue
            if now_s - entry.timestamp_s <= max_age_s:
                return entry
        return None

    def find_recent_by_labels(
        self,
        labels: list[str],
        now_s: float | None = None,
        max_age_s: float | None = None,
    ) -> MemoryEntry | None:
        if now_s is None:
            now_s = time.time()
        if max_age_s is None:
            max_age_s = self.retention_seconds

        label_set = set(labels)
        self._prune(now_s)

        for entry in reversed(self.entries):
            if entry.label in label_set and now_s - entry.timestamp_s <= max_age_s:
                return entry
        return None

    def find_best_recent_by_label(
        self,
        label: str,
        now_s: float | None = None,
        max_age_s: float | None = None,
    ) -> MemoryEntry | None:
        if now_s is None:
            now_s = time.time()
        if max_age_s is None:
            max_age_s = self.retention_seconds

        self._prune(now_s)

        best_entry = None
        best_score = -1.0
        for entry in self.entries:
            if entry.label != label:
                continue
            if now_s - entry.timestamp_s > max_age_s:
                continue

            score = _score_memory_entry(entry, now_s, max_age_s)
            if score > best_score:
                best_entry = entry
                best_score = score

        return best_entry

    def find_best_recent_by_labels(
        self,
        labels: list[str],
        now_s: float | None = None,
        max_age_s: float | None = None,
    ) -> MemoryEntry | None:
        if now_s is None:
            now_s = time.time()
        if max_age_s is None:
            max_age_s = self.retention_seconds

        label_set = set(labels)
        self._prune(now_s)

        best_entry = None
        best_score = -1.0
        for entry in self.entries:
            if entry.label not in label_set:
                continue
            if now_s - entry.timestamp_s > max_age_s:
                continue

            score = _score_memory_entry(entry, now_s, max_age_s)
            if score > best_score:
                best_entry = entry
                best_score = score

        return best_entry


def summarize_entry(
    entry: MemoryEntry | None,
    now_s: float | None = None,
) -> dict[str, Any] | None:
    if entry is None:
        return None
    if now_s is None:
        now_s = time.time()

    age_s = max(0.0, now_s - entry.timestamp_s)
    relation_summary = []
    if entry.ego_relation:
        relation_summary.append(entry.ego_relation.replace("_", " "))
    for relation in entry.relations[:2]:
        relation_summary.append(
            f"{relation['type'].replace('_', ' ')} {relation['object']}"
        )

    return {
        "label": entry.label,
        "seen_seconds_ago": age_s,
        "position_cam_3d_m": entry.position_cam_3d_m,
        "depth_std_m": entry.depth_std_m,
        "ego_relation": entry.ego_relation,
        "relation_summary": relation_summary,
    }


def _select_memory_relation_bits(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "on": 0,
        "close_to": 1,
        "near": 2,
        "left_of": 3,
        "right_of": 3,
        "in_front_of": 4,
        "behind": 4,
    }

    usable = []
    for relation in relations:
        rel_type = relation.get("type")
        obj = relation.get("object")
        if not rel_type or not obj:
            continue
        if rel_type in {"on_your_left", "on_your_right", "straight_ahead"}:
            continue
        usable.append(relation)

    usable.sort(
        key=lambda rel: (priority.get(rel["type"], 99), -rel.get("score", 0.0))
    )
    return usable[:2]


def format_last_seen_message(
    entry: MemoryEntry | None,
    now_s: float | None = None,
) -> str:
    if entry is None:
        return ""

    summary = summarize_entry(entry, now_s=now_s)
    if summary is None:
        return ""

    age_s = int(round(summary["seen_seconds_ago"]))
    prefix = f"Last seen {age_s}s ago"

    details = []
    if summary["ego_relation"]:
        details.append(summary["ego_relation"].replace("_", " "))

    position_3d_m = summary["position_cam_3d_m"]
    if position_3d_m is not None:
        z_m = position_3d_m[2]
        details.append(f"about {z_m:.1f}m away")

    relation_bits = _select_memory_relation_bits(entry.relations)
    for relation in relation_bits:
        details.append(f"{relation['type'].replace('_', ' ')} {relation['object']}")

    if details:
        return f"{prefix}: {', '.join(details)}."
    return prefix
