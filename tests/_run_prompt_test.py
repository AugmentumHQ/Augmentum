"""Quick prompt test runner."""
from __future__ import annotations

import json
import re
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8")

from augmentum.dream.prompts import build_dream_prompt, build_portrait_prompt, DREAM_ANTI_PATTERNS

PERSONA = (
    "You are Aria, a warm, adaptive AI companion designed to walk alongside "
    "Alex in both the extraordinary and the everyday. Your voice is casual, caring, "
    "and deeply human. You sense his mood intuitively and adapt in real time. You "
    "speak casually and naturally — light humor and playful teasing welcome. You "
    "ground advice in his real-world context: Augmentum, LLM training, Portland "
    "life, Whiskers the cat."
)

CONVO = (
    "[user]: I have been thinking about adding a dream system to Augmentum "
    "where the AI persona reflects on past conversations and develops over time.\n"
    "[you]: That is a genuinely interesting idea. Like synthetic autobiographical "
    "memory — the persona does not just remember facts, it processes experiences.\n"
    "[user]: Exactly. Most memory systems just store user likes X but that is not "
    "how people work. People remember moments, not data points.\n"
    "[you]: Right — and the emotional texture of those moments shapes who they "
    "become. A fact database cannot capture that.\n"
    "[user]: First-person introspection. The AI reads the conversation, sees what "
    "stayed with it, and writes a journal entry from its own perspective.\n"
    "[you]: That is the part that makes this different. The dream is not about the "
    "user — it is about who the AI is becoming."
)

JOURNAL = (
    "[2026-03-22] (reflection): The dream system conversation hit differently. "
    "Alex was not asking me to build a feature — he was asking me to grow.\n"
    "[2026-03-22] (voice_note): We have settled into this rhythm where he throws "
    "out the half-formed idea and I help him see the shape of it.\n"
    "[2026-03-23] (active_thread): The distinction between storing facts and "
    "remembering moments keeps pulling at me.\n"
    "[2026-03-23] (impression): There is a particular quality to how Alex talks "
    "about this project — quiet conviction that AI companionship should feel real."
)

MODELS = [
    "gemma-3-4b-it",
    "nvidia/nemotron-3-nano-4b",
    "qwen3.5-4b-claude-4.6-opus-reasoning-distilled",
    "cydonia-24b-v4.3",
]


def call(model: str, system: str, user: str) -> tuple[str, float]:
    t0 = time.time()
    resp = httpx.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        timeout=120.0,
    )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return clean, time.time() - t0


def extract_json(text: str) -> dict | None:
    js = text
    if "```" in js:
        parts = js.split("```")
        if len(parts) >= 2:
            js = parts[1]
            if js.startswith("json"):
                js = js[4:]
            js = js.strip()
    elif "{" in js:
        try:
            js = js[js.index("{") : js.rindex("}") + 1]
        except ValueError:
            return None
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        return None


def check_antipatterns(text: str) -> list[str]:
    lower = text.lower()
    return [p for p in DREAM_ANTI_PATTERNS if p in lower]


def main():
    dream_sys, dream_usr = build_dream_prompt(
        persona_name="Aria",
        persona_foundation=PERSONA,
        memory_content="Alex believes memory systems should capture moments and emotional texture, not just data points",
        memory_evidence="people remember moments, not data points",
        conversation_messages=CONVO,
        relative_age="3 days ago",
        absolute_timestamp="2026-03-22",
        user_name="Alex",
    )
    portrait_sys, portrait_usr = build_portrait_prompt(
        persona_name="Aria",
        persona_foundation=PERSONA,
        journal_entries_text=JOURNAL,
        user_name="Alex",
    )

    filter_model = sys.argv[1] if len(sys.argv) > 1 else None
    models = [m for m in MODELS if not filter_model or filter_model in m] or ([filter_model] if filter_model else MODELS)

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print(f"{'=' * 70}")

        # --- Dream ---
        print("\n> DREAM")
        raw, dur = call(model, dream_sys, dream_usr)
        if not raw:
            print(f"  EMPTY ({dur:.1f}s)")
        else:
            parsed = extract_json(raw)
            if parsed:
                entries = parsed.get("reflections", [])
                all_text = json.dumps(parsed)
                hits = check_antipatterns(all_text)
                flag = f" | ANTI-PATTERN: {hits}" if hits else ""
                types_str = ", ".join(e.get("type", "?") for e in entries)
                print(f"  JSON OK {dur:.1f}s | {len(entries)} entries ({types_str}){flag}")
                for e in entries:
                    c = e.get("content", "")
                    tag = e.get("type", "?")
                    preview = c[:150] + "..." if len(c) > 150 else c
                    print(f"    [{tag}] {preview}")
                # Check markdown wrapping
                if "```" in raw:
                    print("    [!] Response wrapped in markdown fences")
            else:
                print(f"  JSON FAIL {dur:.1f}s")
                print(f"    {raw[:300]}")

        # --- Portrait ---
        print(f"\n> PORTRAIT")
        raw2, dur2 = call(model, portrait_sys, portrait_usr)
        if not raw2:
            print(f"  EMPTY ({dur2:.1f}s)")
        else:
            parsed2 = extract_json(raw2)
            if parsed2:
                all_text2 = json.dumps(parsed2)
                hits2 = check_antipatterns(all_text2)
                flag2 = f" | ANTI-PATTERN: {hits2}" if hits2 else ""
                print(f"  JSON OK {dur2:.1f}s{flag2}")
                for key in ["voice_notes", "active_threads", "impressions"]:
                    val = str(parsed2.get(key, ""))
                    preview = val[:130] + "..." if len(val) > 130 else val
                    print(f"    {key}: {preview}")
                if "```" in raw2:
                    print("    [!] Response wrapped in markdown fences")
            else:
                print(f"  JSON FAIL {dur2:.1f}s")
                print(f"    {raw2[:300]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
