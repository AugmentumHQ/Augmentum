"""Shared /v command detection and direct image generation.

Used by all handler modes (narrative, passthrough, analytical) to detect
the ``/v`` prefix in the last user message and optionally generate images.
"""

from __future__ import annotations

from augmentum.models.base import InternalChatRequest, Message


def extract_v_command(
    request: InternalChatRequest,
    fallback_text: str = "Continue the scene.",
) -> tuple[bool, str, InternalChatRequest]:
    """Check for /v command in the last user message.

    Scans backward from the last message to find the latest user message.
    If it starts with ``/v``, strips the command and returns a cleaned request.

    Args:
        request: The incoming chat request.
        fallback_text: Text to substitute when ``/v`` has no instruction.

    Returns:
        ``(has_v_command, user_instruction, cleaned_request)``
    """
    for i in range(len(request.messages) - 1, -1, -1):
        msg = request.messages[i]
        if msg.role == "user":
            content = msg.content.strip()
            if content.startswith("/v"):
                instruction = content[2:].strip()
                new_messages = list(request.messages)
                new_messages[i] = Message(
                    role=msg.role,
                    content=instruction if instruction else fallback_text,
                    images=msg.images,
                    tool_calls=msg.tool_calls,
                )
                cleaned = InternalChatRequest(
                    model=request.model,
                    messages=new_messages,
                    stream=request.stream,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    seed=request.seed,
                    tools=request.tools,
                    format=request.format,
                    keep_alive=request.keep_alive,
                    raw_options=request.raw_options,
                )
                return True, instruction, cleaned
            break
    return False, "", request


async def generate_direct_image(
    instruction: str,
    image_queue,
    session_id: str,
    *,
    user_id: str = "",
) -> str | None:
    """Generate an image using the instruction text directly as the SD prompt.

    For passthrough/analytical modes — no distiller LLM call, just sends
    the user's text straight to the image backend.

    ``user_id`` is required for multi-tenant ownership — without it the
    resulting image_generations row is orphaned and cannot be deleted from
    the UI. Callers should pass ``self._user_id`` from the handler.

    Returns:
        The image URL path (e.g. ``/api/image/{image_id}``) or ``None`` on failure.
    """
    from augmentum.config import settings
    from augmentum.image.queue import GenerationJob
    from augmentum.utils.logging import get_logger

    log = get_logger(__name__)

    prompt = instruction.strip() if instruction.strip() else "a scene"
    negative = "blurry, low quality, deformed, ugly, watermark, text, words, letters"

    try:
        job = GenerationJob(
            prompt=prompt,
            negative_prompt=negative,
            model=settings.image_default_model,
            width=settings.image_default_width,
            height=settings.image_default_height,
            steps=settings.image_default_steps,
            cfg_scale=settings.image_default_cfg,
            session_id=session_id,
            user_id=user_id,
        )
        job = await image_queue.submit(job)
        result = await image_queue.wait_for_result(job, timeout=300.0)
        return f"/api/image/{result['image_id']}"
    except Exception:
        log.warning("direct_image_generation_failed", exc_info=True)
        return None
