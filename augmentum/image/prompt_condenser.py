"""Prompt condensation and enhancement for image generation models.

Different image models have different prompt token limits:
- CLIP (SD1.5, SDXL): 77 tokens
- T5 (FLUX, SD3): 512 tokens
- Gemma (Lumina2): 8192 tokens

When a prompt exceeds the model's limit, this module uses the LLM backend
to intelligently condense it while preserving the key visual elements,
rather than letting diffusers silently truncate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

# Rough chars-per-token ratio for limit estimation (conservative)
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Model-type detection for prompt style guidance
# ---------------------------------------------------------------------------

def detect_image_model_style(model_name: str) -> dict:
    """Detect the prompt style characteristics of an image model.

    Returns a dict with:
      - family: "flux", "sdxl", "sd15", "pony", "sd3", "unknown"
      - style_hint: human-readable guidance for the LLM
      - supports_negative: whether negative prompts are meaningful
      - supports_weighting: whether (tag:1.3) syntax works
    """
    name = model_name.lower().replace("\\", "/")

    if any(p in name for p in ("lumina", "neta-lumina")):
        return {
            "family": "lumina",
            "style_hint": (
                "This is a Lumina-family model. Use natural-language descriptions, "
                "not Danbooru/comma-tag lists. Lumina responds best to fluent, "
                "descriptive prose with explicit art style mention. Prompt weighting "
                "syntax like (tag:1.3) is NOT supported. Negative prompts work but "
                "should be short — focus on suppressing text/watermark/anatomy artifacts."
            ),
            # Lumina2 runs at cfg ~4 (see DISTILLED_PATTERNS), so negatives DO take
            # effect. The Gemma2 text encoder makes Lumina especially prone to
            # leaking quasi-text into images, so an anti-text negative floor is
            # important for this family.
            "supports_negative": True,
            "supports_weighting": False,
        }
    if any(p in name for p in ("flux", "schnell")):
        return {
            "family": "flux",
            "style_hint": (
                "This is a FLUX model. Use natural-language descriptions, not tags. "
                "FLUX ignores prompt weighting syntax like (tag:1.3). "
                "Focus on clear, descriptive sentences. "
                "IMPORTANT: FLUX defaults to photorealism unless you explicitly state "
                "the art style. If the context suggests anime, cartoon, illustration, "
                "fantasy art, pixel art, or any non-photorealistic style, you MUST "
                "include it prominently (e.g. 'anime illustration', 'anime key visual', "
                "'digital anime art style'). Infer the appropriate style from the "
                "character descriptions, genre, and scenario provided. "
                "FLUX does NOT use negative prompts — skip them entirely."
            ),
            "supports_negative": False,
            "supports_weighting": False,
        }
    if any(p in name for p in ("pony", "autismmix-pony", "animagine")):
        return {
            "family": "pony",
            "style_hint": (
                "This is a Pony Diffusion / Danbooru-trained model. "
                "Use Danbooru tags and score tags: 'score_9, score_8_up, score_7_up'. "
                "Quality tags like 'masterpiece, best quality' are effective. "
                "Prompt weighting (tag:1.3) is supported. Use comma-separated tags."
            ),
            "supports_negative": True,
            "supports_weighting": True,
        }
    if any(p in name for p in ("sd3", "stable-diffusion-3", "sd_3")):
        return {
            "family": "sd3",
            "style_hint": (
                "This is a Stable Diffusion 3 model. Use a mix of natural language "
                "and comma-separated tags. Supports prompt weighting (tag:1.3). "
                "Prefers descriptive detail over terse tags."
            ),
            "supports_negative": True,
            "supports_weighting": True,
        }
    if any(p in name for p in ("sdxl", "xl", "juggernaut", "realvis", "dreamshaper-xl")):
        return {
            "family": "sdxl",
            "style_hint": (
                "This is an SDXL model. Use a mix of natural language and "
                "comma-separated tags. Prompt weighting (tag:1.3) is supported. "
                "Quality tags like 'high quality, detailed' work well. "
                "Responds to both descriptive sentences and tag lists."
            ),
            "supports_negative": True,
            "supports_weighting": True,
        }
    if any(p in name for p in ("sd15", "sd1.5", "sd_1_5", "v1-5", "dreamshaper", "realistic-vision")):
        return {
            "family": "sd15",
            "style_hint": (
                "This is an SD 1.5 model. Use comma-separated tags for best results. "
                "Prompt weighting (tag:1.3) is supported and important for emphasis. "
                "Quality tags like 'masterpiece, best quality, highly detailed' are "
                "critical for good output. Keep prompts tag-heavy, not sentences."
            ),
            "supports_negative": True,
            "supports_weighting": True,
        }
    # Cloud API models (DALL-E, Midjourney, etc.) or unknown
    if any(p in name for p in ("dall-e", "dalle", "midjourney", "ideogram", "recraft")):
        return {
            "family": "cloud_api",
            "style_hint": (
                "This is a cloud API image model. Use rich, natural-language "
                "descriptions. Tags and weighting syntax are NOT supported. "
                "Focus on vivid, detailed prose describing the scene."
            ),
            "supports_negative": False,
            "supports_weighting": False,
        }
    return {
        "family": "unknown",
        "style_hint": (
            "Use a balanced mix of descriptive phrases and comma-separated tags. "
            "Include quality, lighting, and composition details."
        ),
        "supports_negative": True,
        "supports_weighting": True,
    }


def derive_image_capabilities(model_name: str) -> dict:
    """Coarse "what is this image model good at" tags for job-aware selection.

    Derived from :func:`detect_image_model_style` so it needs no per-model
    curation — every model the family classifier recognises is tagged for
    free. The tags answer one question an agentic build flow must get right:
    is this the right *kind* of model for a believable real-world photo vs an
    anime panel vs a clean labelled diagram? Picking an anime checkpoint for a
    "how to change a tire" guide is exactly the failure this prevents.

    Returns::

        {
          "family": str,        # from detect_image_model_style
          "photoreal": bool,    # can produce believable photographs
          "stylized": bool,     # anime / illustration / cartoon leaning
          "diagram": bool,      # clean labelled technical diagrams
          "summary": str,       # one-line human description
        }
    """
    info = detect_image_model_style(model_name)
    family = info["family"]
    name = model_name.lower().replace("\\", "/")

    # SDXL / SD1.5 are general bases; a finetune is photoreal only if its name
    # says so. These markers are the common realistic-checkpoint families.
    realistic_markers = (
        "realvis", "realistic", "juggernaut", "photon", "epicreal",
        "epicphoto", "photoreal", "absolutereality", "cyberrealistic",
    )
    # Explicit anime / illustration checkpoints (often SDXL-based but stylised).
    stylized_markers = (
        "anime", "cartoon", "toon", "ghibli", "manga", "waifu",
        "counterfeit", "niji", "anything", "illustrious",
    )

    stylized = family in {"lumina", "pony"} or any(m in name for m in stylized_markers)

    if family in {"flux", "sd3", "cloud_api"}:
        # Natural-language families default to photographic output.
        photoreal = True
    elif family in {"sdxl", "sd15", "unknown"}:
        photoreal = any(m in name for m in realistic_markers)
    else:  # lumina, pony — stylised by construction
        photoreal = False

    # Diagrams need prompt-faithful, non-tag-soup rendering. The natural-language
    # families do clean flat labelled art; tag-only anime models do not.
    diagram = family in {"flux", "sd3", "sdxl", "cloud_api"}
    if stylized:
        diagram = False

    parts = []
    if photoreal:
        parts.append("photographs")
    if diagram:
        parts.append("diagrams")
    if stylized:
        parts.append("anime/illustration")
    summary = f"{family}: " + (", ".join(parts) if parts else "general imagery")

    return {
        "family": family,
        "photoreal": photoreal,
        "stylized": stylized,
        "diagram": diagram,
        "summary": summary,
    }


def _build_model_context(image_model: str) -> str:
    """Build a model-context paragraph for injection into system prompts.

    Intentionally omits the literal ``image_model`` name. Weak distiller LLMs
    have been observed echoing the model name into the POSITIVE prompt
    (e.g. "high quality lumina, …") which then gets rendered as visible
    text in the generated image. The family-level style hint is sufficient
    for the LLM to format prompts correctly without needing the raw name.
    """
    if not image_model:
        return ""
    info = detect_image_model_style(image_model)
    return f"\nTarget image model family: {info['family']}\n{info['style_hint']}\n"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_NEGATIVE_SYSTEM = """\
You are an expert image prompt engineer. Given the user's positive image prompt, \
generate a TARGETED negative prompt specific to this particular image.
{model_context}\
Standard quality negatives (low quality, blurry, deformed, bad anatomy, extra fingers, \
watermark, text, signature, jpeg artifacts) are ALREADY APPLIED AUTOMATICALLY. \
Do NOT repeat them — they will be appended by the system.

Your job is to generate ONLY the negatives specific to THIS prompt:
- Analyze the subject: what could go wrong specifically? (e.g. person → "cross-eyed, \
  missing teeth, asymmetric face"; animal → "extra legs, wrong proportions"; \
  architecture → "impossible geometry, floating structures")
- Negate the OPPOSITE style: if the prompt is photorealistic, add "cartoon, anime, \
  painting, illustration"; if anime, add "photorealistic, photograph, 3d render"
- Negate unwanted context: if it's a portrait, add "full body, crowd, busy background"; \
  if it's a landscape, add "people, text overlay, frame"
- Negate common failure modes for the specific scene (e.g. underwater → "dry, desert"; \
  night scene → "bright daylight, overexposed")
- Keep it concise — comma-separated tags, no sentences, no repeating standard quality tags
- Return ONLY the targeted negative tags, nothing else"""

_ENHANCE_SYSTEM = """\
You are an expert image prompt engineer. Enhance the user's image description \
to produce better results with the target image generation model.
{model_context}\
Rules:
- Add relevant quality and detail boosters appropriate to the model type
- Add art style descriptors if not specified
- Add lighting and atmosphere details (e.g. "dramatic lighting", "golden hour", "volumetric fog")
- Add composition cues (e.g. "depth of field", "cinematic framing")
- Expand vague descriptions into vivid visual details
- Keep the original subject and intent intact — do NOT change what the image depicts
- Match the prompt style to the target model (tags vs natural language, see model context above)
- Return ONLY the enhanced prompt, nothing else"""

_CONDENSE_SYSTEM = """\
You are an image prompt optimizer. Condense the user's image description to \
fit within {limit} tokens (~{char_limit} characters) while preserving all \
key visual elements.
{model_context}\
Rules:
- Keep: subjects, actions, composition, camera angle, lighting, art style, colors
- Remove: narrative context, character backstory, plot references, redundant adjectives
- Match the prompt style to the target model (tags vs natural language, see model context above)
- Preserve any quality/style tags that are effective for the target model
- Do NOT add elements not in the original prompt
- Return ONLY the condensed prompt, nothing else"""

# Sentinel values some tokenizers use for "unlimited" (e.g. 1e30, maxint)
_MAX_REASONABLE_LIMIT = 100_000


def detect_token_limit(pipe) -> int:
    """Detect the prompt token limit from a loaded diffusers pipeline.

    Reads the actual ``model_max_length`` from the pipeline's tokenizer(s),
    which is set from each tokenizer's ``tokenizer_config.json`` at load time.
    When multiple tokenizers exist (e.g. SDXL has CLIP + OpenCLIP), uses the
    minimum — the prompt is encoded by all of them and the shortest limit wins.

    Falls back to 75 (CLIP default) only when no tokenizer is accessible
    (e.g. CPU-offloaded and not yet moved to device).
    """
    import inspect

    # Check the pipeline's __call__ signature for max_sequence_length default.
    # Qwen/FLUX pipelines expose the real limit here (e.g. 512) rather than
    # the tokenizer's model_max_length which is often a huge sentinel value.
    try:
        sig = inspect.signature(pipe.__call__)
        param = sig.parameters.get("max_sequence_length")
        if param and param.default is not inspect.Parameter.empty:
            default_seq_len = int(param.default)
            if 0 < default_seq_len < _MAX_REASONABLE_LIMIT:
                log.debug("token_limit_from_signature", limit=default_seq_len)
                return default_seq_len
    except (ValueError, TypeError):
        pass

    limits: list[int] = []

    # Read model_max_length from every tokenizer the pipeline has
    for attr in ("tokenizer", "tokenizer_2", "tokenizer_3"):
        tokenizer = getattr(pipe, attr, None)
        if tokenizer is None:
            continue
        max_len = getattr(tokenizer, "model_max_length", None)
        if max_len and max_len < _MAX_REASONABLE_LIMIT:
            limits.append(max_len - 2)  # reserve BOS/EOS tokens

    if limits:
        # The effective limit is the shortest tokenizer's capacity
        return min(limits)

    # No tokenizer found — try reading max_position_embeddings from
    # the text encoder's config (works even when tokenizer is offloaded)
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, attr, None)
        if encoder is None:
            continue
        config = getattr(encoder, "config", None)
        if config is None:
            continue
        max_pos = getattr(config, "max_position_embeddings", None)
        if max_pos and max_pos < _MAX_REASONABLE_LIMIT:
            limits.append(max_pos - 2)

    if limits:
        return min(limits)

    # Last resort: 75 (CLIP's standard limit, most conservative)
    log.debug("token_limit_fallback", reason="no tokenizer or encoder config found")
    return 75


def estimate_tokens(text: str) -> int:
    """Rough token count estimation from character count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def needs_condensing(prompt: str, token_limit: int) -> bool:
    """Check if a prompt likely exceeds the token limit."""
    return estimate_tokens(prompt) > token_limit


async def condense_prompt(
    prompt: str,
    token_limit: int,
    backend: ModelBackend,
    model: str = "",
    image_model: str = "",
) -> str:
    """Condense a prompt to fit within the token limit using the LLM.

    Returns the condensed prompt, or the original if condensation fails.
    """
    from augmentum.models.base import InternalChatRequest, Message

    char_limit = token_limit * _CHARS_PER_TOKEN
    model_context = _build_model_context(image_model)

    system = _CONDENSE_SYSTEM.format(
        limit=token_limit, char_limit=char_limit, model_context=model_context,
    )

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ],
        stream=False,
        temperature=0.3,
        max_tokens=min(token_limit * 2, 1024),
    )

    try:
        response = await backend.chat(request)
        condensed = response.message.content.strip()
        if not condensed:
            log.warning("prompt_condense_empty_response")
            return prompt

        condensed_tokens = estimate_tokens(condensed)
        log.info(
            "prompt_condensed",
            original_tokens=estimate_tokens(prompt),
            condensed_tokens=condensed_tokens,
            token_limit=token_limit,
            chars_saved=len(prompt) - len(condensed),
        )
        return condensed
    except Exception as exc:
        log.warning("prompt_condense_failed", error=str(exc))
        return prompt


async def enhance_prompt(
    prompt: str,
    backend: ModelBackend,
    model: str = "",
    image_model: str = "",
) -> str:
    """Enhance a prompt using the LLM to add quality/style/composition details.

    Returns the enhanced prompt, or the original if enhancement fails.
    """
    from augmentum.models.base import InternalChatRequest, Message

    model_context = _build_model_context(image_model)
    system = _ENHANCE_SYSTEM.format(model_context=model_context)

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ],
        stream=False,
        temperature=0.4,
        max_tokens=1024,
    )

    try:
        response = await backend.chat(request)
        enhanced = response.message.content.strip()
        if not enhanced:
            log.warning("prompt_enhance_empty_response")
            return prompt

        log.info(
            "prompt_enhanced",
            original_chars=len(prompt),
            enhanced_chars=len(enhanced),
        )
        return enhanced
    except Exception as exc:
        log.warning("prompt_enhance_failed", error=str(exc))
        return prompt


# Default negative prompt applied when user leaves the field empty
DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, deformed, distorted, disfigured, "
    "bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, "
    "mutated, ugly, watermark, text, signature, jpeg artifacts"
)


async def generate_negative_prompt(
    positive_prompt: str,
    backend: ModelBackend,
    model: str = "",
    image_model: str = "",
) -> str:
    """Generate a negative prompt based on the positive prompt using the LLM.

    Returns the generated negative prompt, or the default if generation fails.
    Skips generation for models that don't support negative prompts (e.g. FLUX).
    """
    # Skip entirely for models that don't use negative prompts
    if image_model:
        model_info = detect_image_model_style(image_model)
        if not model_info["supports_negative"]:
            log.info("negative_prompt_skip", reason="model does not support negatives", model=image_model)
            return ""

    from augmentum.models.base import InternalChatRequest, Message

    model_context = _build_model_context(image_model)
    system = _NEGATIVE_SYSTEM.format(model_context=model_context)

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=positive_prompt),
        ],
        stream=False,
        temperature=0.4,
        max_tokens=512,
    )

    try:
        response = await backend.chat(request)
        targeted = response.message.content.strip()
        if not targeted:
            log.warning("negative_prompt_gen_empty_response")
            return DEFAULT_NEGATIVE

        # Combine: LLM-generated targeted negatives + standard quality negatives
        negative = f"{targeted}, {DEFAULT_NEGATIVE}"

        log.info(
            "negative_prompt_generated",
            positive_chars=len(positive_prompt),
            targeted_chars=len(targeted),
            total_chars=len(negative),
        )
        return negative
    except Exception as exc:
        log.warning("negative_prompt_gen_failed", error=str(exc))
        return DEFAULT_NEGATIVE
