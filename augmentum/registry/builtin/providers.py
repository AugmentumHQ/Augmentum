"""External provider API keys + base URLs.

These are sensitive — every key is admin_only, never companion-
surfaceable, and the in-process Settings dataclass + persistent
settings_store encrypt them at rest. The registry's purpose for these
is structural: they have labels and descriptions for the discovery
UI, but the actual key management surface continues to live in the
Providers tab (which uses the existing managed-services pipeline).
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_TAGS_KEY = ("provider", "secret")
_TAGS_URL = ("provider", "advanced")


def _register_api_key(
    r: SettingsRegistry, key: str, label: str, provider: str
) -> None:
    r.register(
        Setting(
            key=key,
            kind="str",
            default="",
            label=label,
            description=(
                f"API key for the {provider} provider. Stored encrypted at "
                f"rest. Empty = provider disabled."
            ),
            section=f"providers.{provider.lower().replace(' ', '_')}",
            max_length=512,
            tags=_TAGS_KEY,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )


def _register_base_url(
    r: SettingsRegistry, key: str, label: str, provider: str, default: str = ""
) -> None:
    r.register(
        Setting(
            key=key,
            kind="str",
            default=default,
            label=label,
            description=(
                f"Base URL for the {provider} provider. Override only for "
                f"private endpoints or on-prem mirrors."
            ),
            section=f"providers.{provider.lower().replace(' ', '_')}",
            max_length=256,
            tags=_TAGS_URL,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )


def register(r: SettingsRegistry) -> None:
    # ============== Cloud-model API keys ==============
    _register_api_key(r, "anthropic_api_key", "Anthropic API key", "Anthropic")
    _register_base_url(
        r,
        "anthropic_base_url",
        "Anthropic base URL",
        "Anthropic",
        default="https://api.anthropic.com/v1",
    )

    _register_api_key(r, "openrouter_api_key", "OpenRouter API key", "OpenRouter")
    _register_api_key(r, "deepseek_api_key", "DeepSeek API key", "DeepSeek")
    _register_api_key(r, "fireworks_api_key", "Fireworks API key", "Fireworks")
    _register_api_key(r, "groq_api_key", "Groq API key", "Groq")
    _register_api_key(r, "mistral_api_key", "Mistral API key", "Mistral")
    _register_api_key(r, "perplexity_api_key", "Perplexity API key", "Perplexity")
    _register_api_key(r, "xai_api_key", "xAI API key", "xAI")
    _register_api_key(r, "cohere_api_key", "Cohere API key", "Cohere")
    _register_api_key(r, "google_api_key", "Google AI Studio API key", "Google")

    # Azure has multiple fields.
    _register_api_key(r, "azure_api_key", "Azure API key", "Azure")
    r.register(
        Setting(
            key="azure_base_url",
            kind="str",
            default="",
            label="Azure endpoint URL",
            description=(
                "Azure OpenAI endpoint URL "
                "(e.g. https://your-resource.openai.azure.com)."
            ),
            section="providers.azure",
            max_length=256,
            tags=_TAGS_URL,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="azure_api_version",
            kind="str",
            default="2024-02-01",  # matches augmentum/config.py
            label="Azure API version",
            description=(
                "Azure OpenAI API version pin."
            ),
            section="providers.azure",
            max_length=64,
            tags=_TAGS_URL,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="azure_deployment",
            kind="str",
            default="",
            label="Azure deployment name",
            description=(
                "Azure OpenAI deployment name (your model alias)."
            ),
            section="providers.azure",
            max_length=128,
            tags=_TAGS_URL,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    # Google Vertex auth (parallel to google_api_key).
    r.register(
        Setting(
            key="google_vertex",
            kind="bool",
            default=False,
            label="Use Google Vertex AI",
            description=(
                "Use Google Cloud Vertex AI auth instead of an AI Studio "
                "API key. Requires GOOGLE_APPLICATION_CREDENTIALS set in "
                "the environment."
            ),
            section="providers.google",
            tags=("provider", "advanced"),
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="google_vertex_project",
            kind="str",
            default="",
            label="Vertex project ID",
            description=(
                "Google Cloud project ID when google_vertex is enabled."
            ),
            section="providers.google",
            max_length=128,
            tags=("provider", "advanced"),
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="google_vertex_region",
            kind="str",
            default="us-central1",
            label="Vertex region",
            description=(
                "Google Cloud region when google_vertex is enabled."
            ),
            section="providers.google",
            max_length=64,
            tags=("provider", "advanced"),
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    # HuggingFace token (used for gated model downloads).
    r.register(
        Setting(
            key="huggingface_token",
            kind="str",
            default="",
            label="HuggingFace token",
            description=(
                "HuggingFace API token for gated GGUF / model downloads. "
                "Stored encrypted; only used during model pulls."
            ),
            section="providers.huggingface",
            max_length=512,
            tags=_TAGS_KEY,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
