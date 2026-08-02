"""LoRA discovery, loading, and character matching."""

from __future__ import annotations

import json
import os
from pathlib import Path

from augmentum.image.schemas import LoraInfo
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class LoraManager:
    """Manages LoRA adapter discovery and metadata."""

    def __init__(self, model_dir: str) -> None:
        self._lora_dir = os.path.join(model_dir, "loras")
        Path(self._lora_dir).mkdir(parents=True, exist_ok=True)

    @property
    def lora_dir(self) -> str:
        return self._lora_dir

    def discover(self) -> list[LoraInfo]:
        """Scan the LoRA directory for available adapters."""
        loras = []
        if not os.path.exists(self._lora_dir):
            return loras

        for entry in os.scandir(self._lora_dir):
            if entry.is_file() and entry.name.endswith((".safetensors", ".pt", ".bin")):
                info = self._parse_lora(entry.path)
                loras.append(info)
            elif entry.is_dir():
                # Check for adapter_model.safetensors inside
                adapter_path = os.path.join(entry.path, "adapter_model.safetensors")
                if os.path.exists(adapter_path):
                    info = self._parse_lora(entry.path, is_dir=True)
                    loras.append(info)

        return loras

    def get_path(self, name: str) -> str | None:
        """Get the full path for a LoRA by name."""
        # Direct file match
        for ext in (".safetensors", ".pt", ".bin"):
            path = os.path.join(self._lora_dir, name + ext)
            if os.path.exists(path):
                return path
            path = os.path.join(self._lora_dir, name)
            if os.path.exists(path):
                return path

        # Directory match
        dir_path = os.path.join(self._lora_dir, name)
        if os.path.isdir(dir_path):
            return dir_path

        return None

    def match_character(self, character_name: str) -> LoraInfo | None:
        """Find a LoRA that matches a character name (by name or trigger words)."""
        name_lower = character_name.lower()
        for lora in self.discover():
            if name_lower in lora.name.lower():
                return lora
            if any(name_lower in tw.lower() for tw in lora.trigger_words):
                return lora
        return None

    def _parse_lora(self, path: str, is_dir: bool = False) -> LoraInfo:
        """Parse LoRA metadata from file or directory."""
        if is_dir:
            name = os.path.basename(path)
            size_bytes = sum(
                os.path.getsize(os.path.join(path, f))
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
        else:
            name = Path(path).stem
            size_bytes = os.path.getsize(path)

        # Try to load metadata from companion JSON
        trigger_words = []
        base_model = ""
        meta_path = os.path.splitext(path)[0] + ".json" if not is_dir else os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                trigger_words = meta.get("trigger_words", [])
                base_model = meta.get("base_model", "")
            except Exception:
                log.debug("lora_metadata_parse_failed", path=meta_path, exc_info=True)

        return LoraInfo(
            name=name,
            path=path,
            trigger_words=trigger_words,
            size_bytes=size_bytes,
            base_model=base_model,
        )
