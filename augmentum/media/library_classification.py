"""Provider-agnostic classification for remote media libraries/views."""

from __future__ import annotations

from dataclasses import dataclass, field

PRIMARY_ENTITY_TYPES = (
    "Movie",
    "Series",
    "Season",
    "Episode",
    "MusicVideo",
    "Program",
)

GROUP_TO_ENTITY = {
    "movies": "movie",
    "shows": "series",
    "music_videos": "music_video",
    "live_tv": "live_program",
    "collections": "boxset",
    "playlists": "playlist",
}

COLLECTION_TYPE_MAP = {
    "movies": ("movies", "movie"),
    "tvshows": ("shows", "series"),
    "musicvideos": ("music_videos", "music_video"),
    "livetv": ("live_tv", "live_program"),
    "boxsets": ("collections", "boxset"),
    "playlists": ("playlists", "playlist"),
}

SUPPORTED_SURFACE_GROUPS = frozenset({"movies", "shows", "music_videos"})


@dataclass(slots=True)
class ClassifiedLibrary:
    detected_group: str
    detected_primary_entity: str
    detection_confidence: float
    sample_type_counts: dict[str, int] = field(default_factory=dict)
    sample_notes: dict = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return self.detected_group in SUPPORTED_SURFACE_GROUPS

    @property
    def needs_review(self) -> bool:
        return self.detection_confidence < 0.75


def classify_library(
    *,
    collection_type: str = "",
    sample_type_counts: dict[str, int] | None = None,
    view_type: str = "",
) -> ClassifiedLibrary:
    """Classify a provider-native library into Augmentum surface groups."""
    counts = _clean_counts(sample_type_counts or {})
    ct = str(collection_type or "").strip().lower()
    if ct in COLLECTION_TYPE_MAP:
        group, entity = COLLECTION_TYPE_MAP[ct]
        dominant = _dominant_group(counts)
        confidence = 0.95 if not dominant or dominant == group else 0.75
        notes = {"rule": "collection_type"}
        if dominant and dominant != group:
            notes["sample_disagrees"] = dominant
        return ClassifiedLibrary(
            detected_group=group,
            detected_primary_entity=entity,
            detection_confidence=confidence,
            sample_type_counts=counts,
            sample_notes=notes,
        )

    dominant = _dominant_group(counts)
    if dominant:
        confidence = 0.8 if dominant != "mixed" else 0.5
        return ClassifiedLibrary(
            detected_group=dominant,
            detected_primary_entity=GROUP_TO_ENTITY.get(dominant, "other"),
            detection_confidence=confidence,
            sample_type_counts=counts,
            sample_notes={"rule": "sampled_types", "view_type": view_type or ""},
        )

    return ClassifiedLibrary(
        detected_group="other",
        detected_primary_entity="other",
        detection_confidence=0.0,
        sample_type_counts=counts,
        sample_notes={"rule": "unclassified", "view_type": view_type or ""},
    )


def _clean_counts(raw: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in raw.items():
        item_type = str(key or "").strip()
        if item_type not in PRIMARY_ENTITY_TYPES:
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[item_type] = count
    return out


def _dominant_group(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total <= 0:
        return ""

    family_counts = {
        "movies": counts.get("Movie", 0),
        "shows": (
            counts.get("Series", 0)
            + counts.get("Season", 0)
            + counts.get("Episode", 0)
        ),
        "music_videos": counts.get("MusicVideo", 0),
        "live_tv": counts.get("Program", 0),
    }
    group, count = max(family_counts.items(), key=lambda kv: kv[1])
    if count / total >= 0.7:
        return group
    return "mixed"
