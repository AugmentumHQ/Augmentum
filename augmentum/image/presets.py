"""Genre presets for image generation — curated prompt augmentation profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenrePreset:
    """A curated prompt augmentation profile for a specific genre/style."""

    name: str
    display_name: str
    description: str = ""
    positive_tags: str = ""
    negative_tags: str = ""
    recommended_model: str = ""
    cfg_scale: float = 7.0
    steps: int = 20
    sampler: str = ""
    scheduler: str = ""

    def apply(self, prompt: str, negative_prompt: str = "") -> tuple[str, str]:
        """Return (augmented_prompt, augmented_negative) with preset tags applied."""
        parts = [prompt]
        if self.positive_tags:
            parts.append(self.positive_tags)
        augmented_prompt = ", ".join(p for p in parts if p)

        neg_parts = [negative_prompt] if negative_prompt else []
        if self.negative_tags:
            neg_parts.append(self.negative_tags)
        augmented_negative = ", ".join(p for p in neg_parts if p)

        return augmented_prompt, augmented_negative


BUILTIN_PRESETS: dict[str, GenrePreset] = {
    "fantasy_rpg": GenrePreset(
        name="fantasy_rpg",
        display_name="Fantasy RPG",
        description="High fantasy art style with rich detail, medieval atmosphere",
        positive_tags="fantasy art, highly detailed, dramatic lighting, epic, painterly, medieval, magical atmosphere, rich colors",
        negative_tags="modern clothing, cars, technology, phone, blurry, low quality, deformed, ugly, bad anatomy, watermark, text",
        cfg_scale=7.5,
        steps=25,
        sampler="dpm++_2m_karras",
    ),
    "anime": GenrePreset(
        name="anime",
        display_name="Anime",
        description="Japanese anime illustration style",
        positive_tags="anime style, illustration, vibrant colors, clean lines, detailed eyes, cel shading, high quality anime",
        negative_tags="realistic, photographic, 3d render, blurry, low quality, deformed, bad anatomy, watermark, text, extra limbs",
        cfg_scale=7.0,
        steps=20,
        sampler="euler_a",
    ),
    "scifi": GenrePreset(
        name="scifi",
        display_name="Sci-Fi",
        description="Science fiction concept art with futuristic elements",
        positive_tags="science fiction, futuristic, concept art, highly detailed, neon lighting, cyberpunk, sleek design, advanced technology",
        negative_tags="medieval, fantasy, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
        cfg_scale=7.0,
        steps=25,
        sampler="dpm++_2m_sde_karras",
    ),
    "horror": GenrePreset(
        name="horror",
        display_name="Horror",
        description="Dark, atmospheric horror imagery",
        positive_tags="horror art, dark atmosphere, eerie lighting, unsettling, gothic, detailed shadows, muted colors, ominous",
        negative_tags="bright colors, cheerful, cartoon, low quality, blurry, deformed, watermark, text",
        cfg_scale=8.0,
        steps=25,
        sampler="dpm++_2m_sde_karras",
    ),
    "realism": GenrePreset(
        name="realism",
        display_name="Realism",
        description="Photorealistic image generation",
        positive_tags="photorealistic, highly detailed, sharp focus, professional photography, natural lighting, 8k, ultra detailed",
        negative_tags="cartoon, anime, painting, illustration, drawing, low quality, blurry, deformed, ugly, watermark, text",
        cfg_scale=5.0,
        steps=30,
        sampler="dpm++_2m_karras",
    ),
}


class PresetManager:
    """Manages genre presets, including built-in and custom presets."""

    def __init__(self) -> None:
        self._presets: dict[str, GenrePreset] = dict(BUILTIN_PRESETS)

    def get(self, name: str) -> GenrePreset | None:
        return self._presets.get(name)

    def list_presets(self) -> list[GenrePreset]:
        return list(self._presets.values())

    def add(self, preset: GenrePreset) -> None:
        self._presets[preset.name] = preset

    def remove(self, name: str) -> bool:
        if name in BUILTIN_PRESETS:
            return False  # Cannot remove built-in presets
        return self._presets.pop(name, None) is not None
