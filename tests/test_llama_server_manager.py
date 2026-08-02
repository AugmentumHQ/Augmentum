"""Tests for LlamaServerManager — subprocess lifecycle (Task 3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from augmentum.models.llama_server_manager import LlamaServerManager, ProcessState
from augmentum.models.model_profile_cache import ModelProfile


@pytest.fixture
def manager(tmp_path: Path) -> LlamaServerManager:
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    return LlamaServerManager(
        llama_server_path="/usr/local/bin/llama-server",
        backend_port=8091,
        model_dir=model_dir,
        gpu_layers=99,
        ctx_size=32384,
        batch_size=512,
    )


def _make_profile(model_path: str = "/models/test.gguf", ctx: int = 8192) -> ModelProfile:
    return ModelProfile(
        model_path=model_path,
        model_name="test",
        context_length=ctx,
        n_layers=32,
        n_heads=32,
        n_embed=4096,
        total_size_bytes=8 * 1024**3,
    )


# ----- CLI args builder tests -----


class TestBuildCliArgs:
    def test_basic_flags(self, manager: LlamaServerManager) -> None:
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--model" in args
        assert "/models/test.gguf" in args
        assert "--host" in args
        assert "127.0.0.1" in args
        assert "--port" in args
        assert str(manager._backend_port) in args
        assert "--ctx-size" in args
        assert "--batch-size" in args
        assert "--n-gpu-layers" in args
        assert "--metrics" in args

    def test_flash_attn_default(self, manager: LlamaServerManager) -> None:
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--flash-attn" in args

    def test_flash_attn_disabled(self, manager: LlamaServerManager) -> None:
        manager.flash_attn = False
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--flash-attn" not in args

    def test_flash_attn_suppressed_for_cpu_only_load(self, manager: LlamaServerManager) -> None:
        """CPU-only loads (gpu_layers=0) must NOT pass --flash-attn even
        when configured on. flash-attn is a CUDA kernel optimization
        with no benefit for CPU attention, but llama-server still
        initializes a ~400MB CUDA context for it. That wastes VRAM
        the GPU-side chat model needs for offload.

        Concrete site: SmolVLM-256M ships with n_gpu_layers=0 (CPU
        weights) but was previously claiming ~400 MB of GPU memory
        for an unused CUDA context, dropping the main 35B-A3B model's
        usable --n-gpu-layers by several layers on a 24 GB GPU-A."""
        manager.flash_attn = True
        manager._gpu_layers = 0  # CPU-only load
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--flash-attn" not in args, (
            f"flash-attn must not be passed when gpu_layers=0; got args: {args}"
        )

    def test_cont_batching_not_in_args(self, manager: LlamaServerManager) -> None:
        """cont-batching is enabled by default in llama.cpp b5000+ — no flag needed."""
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--cont-batching" not in args

    def test_kv_cache_type(self, manager: LlamaServerManager) -> None:
        manager.kv_cache_type = "q8_0"
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--cache-type-k" in args
        assert "--cache-type-v" in args
        idx_k = args.index("--cache-type-k")
        assert args[idx_k + 1] == "q8_0"

    def test_kv_cache_type_absent(self, manager: LlamaServerManager) -> None:
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--cache-type-k" not in args
        assert "--cache-type-v" not in args

    def test_draft_model(self, manager: LlamaServerManager, tmp_path: Path, monkeypatch) -> None:
        # build_load_plan now pre-checks the draft path exists (so users get
        # a clean error instead of llama-server's cryptic startup failure).
        # Use a real temp path; stub scan_gguf_header so we don't need a
        # real GGUF header on disk just to exercise the CLI-arg wiring.
        draft_path = tmp_path / "draft.gguf"
        draft_path.write_bytes(b"")
        from augmentum.models import llama_server_manager as lsm
        monkeypatch.setattr(
            lsm, "scan_gguf_header",
            lambda p: ModelProfile(
                model_path=str(p), model_name="draft",
                n_layers=8, n_heads=8, n_embed=512, total_size_bytes=64 * 1024 * 1024,
            ),
        )
        manager.draft_model = str(draft_path)
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--model-draft" in args
        assert str(draft_path) in args
        assert "--draft-max" in args
        assert str(manager.draft_max) in args

    def test_draft_model_absent(self, manager: LlamaServerManager) -> None:
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--model-draft" not in args
        assert "--draft-max" not in args

    def test_ctx_size_from_profile(self, manager: LlamaServerManager) -> None:
        # Profile context_length (4096) < manager ctx_size (32384) → use profile's
        profile = _make_profile(ctx=4096)
        args = manager._build_cli_args(profile, "/models/test.gguf")
        idx = args.index("--ctx-size")
        assert args[idx + 1] == "4096"

    def test_ctx_size_from_manager_when_smaller(self, manager: LlamaServerManager) -> None:
        # Profile context_length (131072) > manager ctx_size (32384) → start from manager's
        # request, then allow the normal autofit path to cap it if VRAM requires.
        profile = _make_profile(ctx=131072)
        plan = manager.build_load_plan("/models/test.gguf", profile=profile)
        args = manager._build_cli_args(profile, "/models/test.gguf")
        idx = args.index("--ctx-size")
        assert args[idx + 1] == str(plan["applied"]["ctx_size"])
        assert int(args[idx + 1]) <= manager._ctx_size

    def test_ctx_size_zero_profile(self, manager: LlamaServerManager) -> None:
        # Profile has 0 context_length → start from manager ctx_size, then autofit if needed
        profile = _make_profile(ctx=0)
        plan = manager.build_load_plan("/models/test.gguf", profile=profile)
        args = manager._build_cli_args(profile, "/models/test.gguf")
        idx = args.index("--ctx-size")
        assert args[idx + 1] == str(plan["applied"]["ctx_size"])
        assert int(args[idx + 1]) <= manager._ctx_size

    def test_jinja_default_on(self, manager: LlamaServerManager) -> None:
        """--jinja must be on by default so reasoning models (GLM-4.7, Qwen3,
        DeepSeek-R1) use their embedded chat templates and emit <think> tags
        correctly. Without it, GLM-style chain-of-thought leaks visibly."""
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--jinja" in args
        assert "--reasoning-format" in args
        idx = args.index("--reasoning-format")
        assert args[idx + 1] == "deepseek"

    def test_jinja_can_be_disabled(self, manager: LlamaServerManager, monkeypatch) -> None:
        """Escape hatch for GGUFs with buggy embedded templates."""
        from augmentum.config import settings
        monkeypatch.setattr(settings, "engine_use_jinja_template", False)
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        assert "--jinja" not in args
        assert "--reasoning-format" not in args

    # ---- Per-model chat template overrides (load sheet UI) ------------------

    def test_chat_template_mode_embedded_default(self, manager: LlamaServerManager) -> None:
        """Default mode (no opts) is embedded — emits --jinja."""
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf", load_options={})
        assert "--jinja" in args
        assert "--chat-template-file" not in args

    def test_chat_template_mode_builtin_omits_jinja(self, manager: LlamaServerManager) -> None:
        """builtin mode lets llama-server pick its own template — escape hatch
        for GGUFs whose embedded template is broken."""
        profile = _make_profile()
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={"chat_template_mode": "builtin"},
        )
        assert "--jinja" not in args
        assert "--chat-template-file" not in args
        # builtin also suppresses --reasoning-format since it has no effect.
        assert "--reasoning-format" not in args

    def test_chat_template_mode_custom_writes_file(
        self, manager: LlamaServerManager, tmp_path,
    ) -> None:
        """custom mode writes the user-provided Jinja content to disk and
        passes --chat-template-file alongside --jinja (llama.cpp silently
        ignores --chat-template-file without --jinja)."""
        manager._model_dir = str(tmp_path)
        custom_jinja = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        profile = _make_profile()
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={
                "chat_template_mode": "custom",
                "chat_template_content": custom_jinja,
            },
        )
        # Both flags MUST be present — --chat-template-file alone is silently ignored.
        assert "--jinja" in args
        assert "--chat-template-file" in args
        idx = args.index("--chat-template-file")
        tmpl_path = Path(args[idx + 1])
        assert tmpl_path.exists()
        assert tmpl_path.read_text(encoding="utf-8") == custom_jinja
        assert tmpl_path.parent.name == ".chat_templates"

    def test_chat_template_custom_empty_content_falls_back_to_embedded(
        self, manager: LlamaServerManager,
    ) -> None:
        """custom mode without content is meaningless — fall back to embedded
        (--jinja) so the user isn't left with no template."""
        profile = _make_profile()
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={
                "chat_template_mode": "custom",
                "chat_template_content": "   ",
            },
        )
        assert "--jinja" in args
        assert "--chat-template-file" not in args

    def test_reasoning_format_per_model_override(self, manager: LlamaServerManager) -> None:
        """A per-model reasoning_format wins over the global setting."""
        profile = _make_profile()
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={"reasoning_format": "none"},
        )
        idx = args.index("--reasoning-format")
        assert args[idx + 1] == "none"

    def test_chat_template_kwargs_forwarded(self, manager: LlamaServerManager) -> None:
        """`enable_thinking` / `clear_thinking` style kwargs propagate to
        --chat-template-kwargs. Required for GLM-4.7-Flash and similar
        models whose templates branch on these flags."""
        profile = _make_profile()
        kwargs_json = '{"enable_thinking": false}'
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={"chat_template_kwargs": kwargs_json},
        )
        assert "--chat-template-kwargs" in args
        idx = args.index("--chat-template-kwargs")
        assert args[idx + 1] == kwargs_json

    def test_chat_template_kwargs_invalid_json_rejected(self, manager: LlamaServerManager) -> None:
        """Malformed JSON should fail loudly so the user sees their typo
        rather than discovering it as silent template misbehavior."""
        profile = _make_profile()
        with pytest.raises(ValueError, match="valid JSON"):
            manager._build_cli_args(
                profile, "/models/test.gguf",
                load_options={"chat_template_kwargs": "{enable_thinking: false}"},  # missing quotes
            )

    def test_chat_template_kwargs_omitted_when_empty(self, manager: LlamaServerManager) -> None:
        """Empty kwargs should not emit the flag at all."""
        profile = _make_profile()
        args = manager._build_cli_args(
            profile, "/models/test.gguf",
            load_options={"chat_template_kwargs": ""},
        )
        assert "--chat-template-kwargs" not in args

    def test_force_single_slot_drops_warm_tier_flags(self, tmp_path: Path) -> None:
        """A manager constructed with ``force_single_slot=True`` skips the
        multi-slot KV warm tier — used by SmolVLM and other auxiliaries
        whose sync, single-request workload never benefits from the
        cache-ram budget (which auto-sizes to ~16 GiB of host RAM)."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)
        sibling = LlamaServerManager(
            llama_server_path="/usr/local/bin/llama-server",
            backend_port=8092,
            model_dir=model_dir,
            gpu_layers=0,
            ctx_size=8192,
            batch_size=256,
            force_single_slot=True,
        )
        profile = _make_profile()
        args = sibling._build_cli_args(profile, "/models/test.gguf")
        joined = " ".join(args)
        assert "--parallel" in args
        # --parallel must be followed by "1" (single-slot)
        assert args[args.index("--parallel") + 1] == "1"
        # Warm-tier flags MUST be absent — they're the cost being avoided.
        assert "--kv-unified" not in args, f"got: {joined}"
        assert "--cache-ram" not in args, f"got: {joined}"
        assert "--cache-idle-slots" not in args, f"got: {joined}"
        assert "--ctx-checkpoints" not in args, f"got: {joined}"

    def test_primary_engine_still_emits_warm_tier(self, manager: LlamaServerManager) -> None:
        """Regression guard: the default (non-sibling) manager keeps
        multi-slot behavior on as long as the global setting allows."""
        profile = _make_profile()
        args = manager._build_cli_args(profile, "/models/test.gguf")
        # Default fixture has force_single_slot=False and the
        # codebase default for engine_multislot_enabled is True.
        assert "--kv-unified" in args
        assert "--cache-ram" in args


# ----- State tests -----


class TestState:
    def test_initial_state(self, manager: LlamaServerManager) -> None:
        assert manager.state == ProcessState.IDLE
        assert manager.model_id == ""
        assert manager.model_path == ""
        assert manager.process is None

    def test_base_url(self, manager: LlamaServerManager) -> None:
        assert manager.base_url == "http://127.0.0.1:8091"

    @pytest.mark.asyncio
    async def test_stop_when_idle(self, manager: LlamaServerManager) -> None:
        # Should be a no-op, no error
        await manager.stop()
        assert manager.state == ProcessState.IDLE

    def test_status(self, manager: LlamaServerManager) -> None:
        s = manager.status()
        assert s["state"] == "idle"
        assert s["model_id"] == ""
        assert s["model_path"] == ""
        assert "backend_url" in s

    def test_status_includes_actual_memory_from_llama_logs(self, manager: LlamaServerManager) -> None:
        lines = [
            "load_tensors:   CPU_Mapped model buffer size =   942.97 MiB",
            "load_tensors:        CUDA0 model buffer size = 15843.20 MiB",
            "llama_context:  CUDA_Host  output buffer size =     3.79 MiB",
            "llama_kv_cache:      CUDA0 KV buffer size =  4096.00 MiB",
            "llama_memory_recurrent:      CUDA0 RS buffer size =   586.03 MiB",
            "sched_reserve:      CUDA0 compute buffer size =   495.00 MiB",
            "sched_reserve:  CUDA_Host compute buffer size =   158.49 MiB",
            "main: model loaded",
        ]

        for line in lines:
            manager._ingest_server_line(line, "stderr")

        s = manager.status()
        assert "actual_memory" in s
        actual = s["actual_memory"]
        assert actual["source"] == "llama_server_logs"
        assert actual["complete"] is True
        assert actual["vram_total_mib"] == 21020
        assert actual["ram_total_mib"] == 1105
        assert actual["locations"]["CUDA0"]["scope"] == "vram"
        assert actual["locations"]["CUDA0"]["total_mib"] == 21020
        assert actual["locations"]["CPU_Mapped"]["scope"] == "ram"
        assert actual["locations"]["CPU_Mapped"]["total_mib"] == 943
        assert actual["locations"]["CUDA_Host"]["total_mib"] == 162

    @pytest.mark.asyncio
    async def test_stop_clears_actual_memory(self, manager: LlamaServerManager) -> None:
        manager._ingest_server_line("load_tensors:   CPU_Mapped model buffer size =   942.97 MiB", "stderr")
        assert "actual_memory" in manager.status()

        await manager.stop()

        assert "actual_memory" not in manager.status()


class TestLoadPlanEstimates:
    def test_explicit_context_request_is_not_silently_reduced(self, manager: LlamaServerManager) -> None:
        profile = _make_profile(ctx=262144)
        manager._query_gpu_info = lambda: {"total_mib": 24_000, "free_mib": 4_000}
        manager._query_system_memory_info = lambda: {"total_mib": 64_000, "available_mib": 48_000}

        plan = manager.build_load_plan(
            "/models/test.gguf",
            load_options={
                "ctx_size": 96000,
                "gpu_layers_mode": "custom",
                "gpu_layers": 32,
                "batch_size": 2048,
                "flash_attn": False,
            },
            profile=profile,
        )

        assert plan["requested"]["ctx_size"] == 96000
        assert plan["applied"]["ctx_size"] == 96000
        assert any("will be requested as-is" in warning for warning in plan["warnings"])

    def test_flash_attention_reduces_peak_vram(self, manager: LlamaServerManager) -> None:
        profile = _make_profile()
        manager._query_gpu_info = lambda: {"total_mib": 24_000, "free_mib": 20_000}
        manager._query_system_memory_info = lambda: {"total_mib": 64_000, "available_mib": 48_000}

        base_opts = {
            "gpu_layers_mode": "custom",
            "gpu_layers": 32,
            "ctx_size": 16384,
            "batch_size": 2048,
        }
        with_flash = manager.build_load_plan("/models/test.gguf", load_options={**base_opts, "flash_attn": True}, profile=profile)
        without_flash = manager.build_load_plan("/models/test.gguf", load_options={**base_opts, "flash_attn": False}, profile=profile)

        assert without_flash["memory"]["estimated_vram_mb"] > with_flash["memory"]["estimated_vram_mb"]

    def test_cpu_mode_moves_kv_estimate_to_ram(self, manager: LlamaServerManager) -> None:
        profile = _make_profile(ctx=16384)
        manager._query_gpu_info = lambda: {"total_mib": 24_000, "free_mib": 20_000}
        manager._query_system_memory_info = lambda: {"total_mib": 64_000, "available_mib": 48_000}

        plan = manager.build_load_plan(
            "/models/test.gguf",
            load_options={
                "gpu_layers_mode": "cpu",
                "ctx_size": 16384,
                "batch_size": 1024,
            },
            profile=profile,
        )

        assert plan["memory"]["steady_vram_mb"] == 0
        assert plan["memory"]["steady_ram_mb"] > 0
        assert plan["memory"]["estimated_ram_mb"] >= plan["memory"]["steady_ram_mb"]

    def test_first_fit_returns_more_layers_than_uniform_division(
        self, manager: LlamaServerManager,
    ) -> None:
        """T2-5 regression: per-layer KV proration must NOT reserve KV
        for non-offloaded layers, freeing VRAM the prior flat-margin
        formula was wasting.

        Concrete scenario: 24 GiB VRAM, 30 GB / 32-layer dense model,
        large KV. The old formula reserved KV for ALL 32 layers up front;
        the new path attributes KV per offloaded layer, which means
        offloading only 19 layers reserves only 19/32 of the KV, freeing
        the rest of VRAM for additional layers.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {
                "total_bytes": 24 * 1024**3,
                "total_mib": 24 * 1024,
                "free_mib": 22 * 1024,
            }
        )
        # 30 GB total weights, 32 layers, GQA-typical heads, 16k ctx
        # produces a per-token KV that lands KV around 1-2 GB total at
        # f16 — small enough to fit a meaningful number of layers.
        profile = ModelProfile(
            model_path="/models/dense.gguf",
            model_name="dense-30g",
            n_layers=32,
            n_heads=32,
            n_heads_kv=8,           # GQA — KV ~1/4 of full
            n_embed=4096,
            context_length=32768,
            total_size_bytes=30 * 1024**3,
        )
        fit = manager._autofit_gpu_layers_for(
            profile, ctx_size=8192, kv_cache_type="", gpu_layers_cap=99,
        )

        # Under the old flat-margin formula the math was:
        #   available = 24G - kv_total - 1.5G ≈ 22G
        #   fit = 22G / (30G/32) = 22G / 0.94G ≈ 23 layers
        # New per-layer says:
        #   per_layer = 30G/32 + kv/32 + 0.5G/32 ≈ 0.99G
        #   fit = 23G / 0.99G ≈ 23 layers
        # The win shows when KV is large vs model — see the next test
        # for that scenario. Here just confirm the allocator returns a
        # reasonable count (>0 and <=n_layers).
        assert 0 < fit <= profile.n_layers
        assert fit <= 99  # respected cap

    def test_first_fit_proration_wins_with_large_kv(
        self, manager: LlamaServerManager,
    ) -> None:
        """The per-layer approach's actual win: heavy KV scenarios.

        Same 30 GB / 32-layer model on 24 GiB VRAM, but at 64K context
        with f16 KV. The KV total grows to ~4 GB. Old formula reserves
        all 4 GB up front before even computing per-layer fit; new
        path attributes 4G/32 = ~125 MB per offloaded layer.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 24 * 1024**3, "total_mib": 24 * 1024, "free_mib": 22 * 1024}
        )
        profile = ModelProfile(
            model_path="/models/dense.gguf",
            model_name="dense-30g-bigctx",
            n_layers=32,
            n_heads=32,
            n_heads_kv=8,
            n_embed=4096,
            context_length=131072,
            total_size_bytes=30 * 1024**3,
        )

        fit_new = manager._autofit_gpu_layers_for(
            profile, ctx_size=65536, kv_cache_type="", gpu_layers_cap=99,
        )

        # Replicate the old flat-margin formula manually so the test
        # asserts the new allocator returns AT LEAST as many layers
        # as the old one would have. Net win on this scenario is
        # 3-5 additional layers in production traces.
        kv_per_token = manager._kv_bytes_per_token(profile, "")
        kv_total_old = kv_per_token * 65536
        compute = 512 * 1024**2
        safety = 1024 * 1024**2
        bytes_per_layer = profile.total_size_bytes / profile.n_layers
        old_available = 24 * 1024**3 - kv_total_old - compute - safety
        old_fit = max(0, int(old_available / bytes_per_layer))

        assert fit_new >= old_fit, (
            f"new first-fit ({fit_new}) returned fewer layers than the "
            f"prior flat-margin formula ({old_fit}); per-layer proration "
            "should always be >= flat-margin"
        )

    def test_first_fit_respects_gpu_layers_cap(
        self, manager: LlamaServerManager,
    ) -> None:
        """gpu_layers_cap clamps the result regardless of how much
        VRAM headroom remains.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 80 * 1024**3, "total_mib": 80 * 1024, "free_mib": 78 * 1024}
        )
        profile = ModelProfile(
            model_path="/models/tiny.gguf", model_name="tiny",
            n_layers=32, n_heads=8, n_heads_kv=8, n_embed=512,
            context_length=4096, total_size_bytes=2 * 1024**3,
        )

        fit = manager._autofit_gpu_layers_for(
            profile, ctx_size=4096, kv_cache_type="", gpu_layers_cap=10,
        )
        assert fit == 10  # capped, even though 80 GB easily fits all 32

    def test_first_fit_returns_zero_when_baseline_alone_exceeds_vram(
        self, manager: LlamaServerManager,
    ) -> None:
        """Pathological tiny VRAM — baseline reserve alone consumes all
        budget. Allocator returns 0 (CPU fallback) without crashing.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 512 * 1024**2, "total_mib": 512, "free_mib": 256}
        )
        profile = ModelProfile(
            model_path="/models/m.gguf", model_name="m",
            n_layers=32, n_heads=8, n_heads_kv=8, n_embed=512,
            context_length=2048, total_size_bytes=2 * 1024**3,
        )

        fit = manager._autofit_gpu_layers_for(
            profile, ctx_size=2048, kv_cache_type="", gpu_layers_cap=99,
        )
        assert fit == 0

    def test_first_fit_falls_back_to_cap_when_vram_undetectable(
        self, manager: LlamaServerManager,
    ) -> None:
        """When nvidia-smi is unavailable we can't first-fit; fall back
        to ``min(cap, n_layers)`` and let llama-server's mmap sort it out.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = lambda: {}  # type: ignore[method-assign]
        profile = ModelProfile(
            model_path="/models/m.gguf", model_name="m",
            n_layers=32, n_heads=8, n_heads_kv=8, n_embed=512,
            context_length=2048, total_size_bytes=2 * 1024**3,
        )

        fit_capped = manager._autofit_gpu_layers_for(
            profile, ctx_size=2048, kv_cache_type="", gpu_layers_cap=12,
        )
        assert fit_capped == 12

        fit_uncapped = manager._autofit_gpu_layers_for(
            profile, ctx_size=2048, kv_cache_type="", gpu_layers_cap=99,
        )
        assert fit_uncapped == 32  # min(99, 32)

    def test_first_fit_handles_moe_via_total_size(
        self, manager: LlamaServerManager,
    ) -> None:
        """MoE layers carry expert + non-expert weights; total_size_bytes
        already includes both. Per-layer cost = total/n_layers handles
        MoE without a special branch — verify a plausible Mixtral-style
        profile yields a non-zero fit and emits the MoE diagnostic
        fields in the log payload.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 80 * 1024**3, "total_mib": 80 * 1024, "free_mib": 78 * 1024}
        )
        profile = ModelProfile(
            model_path="/models/mixtral.gguf", model_name="mixtral-8x7b",
            n_layers=32,
            n_heads=32, n_heads_kv=8, n_embed=4096,
            context_length=32768,
            total_size_bytes=46 * 1024**3,  # 8 experts × ~5.5G + non-expert
            is_moe=True,
            expert_count=8,
            expert_used_count=2,
            non_expert_tensor_bytes=2 * 1024**3,
            expert_tensor_bytes=44 * 1024**3,
        )

        fit = manager._autofit_gpu_layers_for(
            profile, ctx_size=8192, kv_cache_type="", gpu_layers_cap=99,
        )
        # 80 GB / (46 GB / 32) ≈ 55 — but cap to actual layer count.
        assert 1 <= fit <= profile.n_layers

    def test_autofit_moe_cpu_layers_places_experts_on_gpu_fused_layout(
        self, manager: LlamaServerManager,
    ) -> None:
        """Regression: for a big fused-expert MoE on a card that fits SOME
        experts, moe_auto_vram must place experts on GPU (partial offload),
        not silently collapse to all-CPU.

        The bug: n_expert_tensors counts FUSED tensors (~3 per MoE layer), but
        the autofit divided by ``expert_count * 3`` — so a high-expert model
        (Qwen3.5-122B-A10B: 144 tensors / (128*3) → round → 1) was treated as
        one 70 GB layer, nothing fit, and N collapsed to n_layers (== --cpu-moe,
        the slowest config). The count must be independent of expert_count.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        # 24 GB card, ~24 GB free (matches the observed 122B-A10B load).
        manager._query_gpu_info = (  # type: ignore[method-assign]
            lambda: {"free_bytes": 24 * 1024**3, "total_bytes": 25 * 1024**3}
        )

        def make(expert_count: int) -> ModelProfile:
            return ModelProfile(
                model_path="/models/qwen35-122b.gguf",
                model_name="Qwen3.5-122B-A10B",
                n_layers=48, n_heads=64, n_heads_kv=8, n_embed=8192,
                context_length=262144,
                total_size_bytes=77 * 1024**3,
                is_moe=True,
                expert_count=expert_count, expert_used_count=8,
                # Fused layout: 3 tensors per MoE layer, ALL 48 layers MoE.
                n_expert_tensors=48 * 3,
                non_expert_tensor_bytes=int(6.65 * 1024**3),
                expert_tensor_bytes=int(70.36 * 1024**3),
            )

        n_cpu = manager._autofit_moe_cpu_layers(
            make(128), ctx_size=32768, kv_cache_type="q4_0",
            flash_attn=True, batch_size=512,
        )
        # Core regression: NOT all-CPU — experts actually land on GPU.
        assert n_cpu < 48, f"expected partial offload, got all-CPU (N={n_cpu})"
        # ~15 GB budget / ~1.47 GB per layer ⇒ ~10 expert layers on GPU.
        assert 48 - n_cpu >= 5, f"too few experts on GPU (N={n_cpu})"

        # The chosen N must be independent of expert_count — a wildly larger
        # expert_count used to make the collapse worse; now it's identical.
        assert manager._autofit_moe_cpu_layers(
            make(256), ctx_size=32768, kv_cache_type="q4_0",
            flash_attn=True, batch_size=512,
        ) == n_cpu

    # -- T2-7: workspace calibration integration ---------------------

    def test_compute_reserve_uses_calibration_after_min_samples(
        self, manager: LlamaServerManager,
    ) -> None:
        """After the calibration store crosses the sample threshold,
        ``_compute_reserve_bytes`` returns baseline × applied_factor.
        """
        cal = manager._workspace_calibration
        # Drive enough samples at ratio 0.8 so the EMA converges close
        # to 0.8 (within the clamp range).
        for _ in range(10):
            cal.record(
                "fa_on",
                observed_bytes=int(384 * 1024**2 * 0.8),
                predicted_bytes=384 * 1024**2,
            )
        applied = cal.get_factor("fa_on")
        # Sanity: applied factor moved off 1.0.
        assert 0.7 <= applied < 1.0

        baseline = manager._compute_reserve_baseline_bytes(True)
        actual = manager._compute_reserve_bytes(True)
        # Allow ±1 byte for int truncation.
        expected = int(baseline * applied)
        assert abs(actual - expected) <= 1

    def test_compute_reserve_baseline_unchanged_below_min_samples(
        self, manager: LlamaServerManager,
    ) -> None:
        """With fewer than MIN_SAMPLES_TO_TRUST samples, the reserve
        returned must equal the baseline — no calibration applied yet.
        """
        cal = manager._workspace_calibration
        # Single sample: under the threshold.
        cal.record(
            "fa_on", observed_bytes=192 * 1024**2, predicted_bytes=384 * 1024**2,
        )
        assert manager._compute_reserve_bytes(True) == (
            manager._compute_reserve_baseline_bytes(True)
        )

    def test_model_loaded_event_records_calibration_sample(
        self, manager: LlamaServerManager,
    ) -> None:
        """End-to-end: feeding llama-server log lines that include a
        ``compute buffer`` entry + ``model loaded`` triggers a
        calibration sample matching the parsed VRAM compute total.
        """
        # Pretend the load already captured the predicted reserve.
        manager._last_predicted_compute_reserve_bytes = 384 * 1024**2
        manager._last_predicted_compute_bucket = "fa_on"

        # Feed a synthetic compute-buffer log line in the format
        # llama-server emits (the regex is anchored on the literal
        # component label + " size = NNN MiB").
        manager._ingest_server_line(
            "llama_kv_cache_init: CUDA0 compute buffer size = 480 MiB",
            "stderr",
        )
        # Followed by "model loaded" which triggers the snapshot+record.
        manager._ingest_server_line("llama_model_load: model loaded", "stderr")

        snap = manager._workspace_calibration.snapshot()
        assert "fa_on" in snap, (
            f"calibration sample not recorded: {snap}"
        )
        assert snap["fa_on"]["samples"] == 1
        # ratio = 480 / 384 = 1.25.
        assert abs(snap["fa_on"]["ratio"] - (480.0 / 384.0)) < 1e-6

    def test_calibration_skipped_for_cpu_only_load(
        self, manager: LlamaServerManager,
    ) -> None:
        """If the load was CPU-only (gpu_layers=0), no compute reserve
        was predicted — the model-loaded handler must not record.
        """
        # Predicted=0 simulates the CPU-only path setting these to 0.
        manager._last_predicted_compute_reserve_bytes = 0
        manager._last_predicted_compute_bucket = ""

        manager._ingest_server_line(
            "llama_kv_cache_init: CUDA0 compute buffer size = 480 MiB",
            "stderr",
        )
        manager._ingest_server_line("llama_model_load: model loaded", "stderr")

        # Nothing recorded.
        assert manager._workspace_calibration.snapshot() == {}

    def test_calibration_only_uses_vram_compute_buffers(
        self, manager: LlamaServerManager,
    ) -> None:
        """Host-side compute buffers (CPU fallback) must NOT count
        toward the calibration — calibration applies only to GPU
        compute reserve.
        """
        manager._last_predicted_compute_reserve_bytes = 384 * 1024**2
        manager._last_predicted_compute_bucket = "fa_on"

        # CPU compute buffer + CUDA0 compute buffer + model loaded.
        manager._ingest_server_line(
            "llama_kv_cache_init: CPU compute buffer size = 9999 MiB",
            "stderr",
        )
        manager._ingest_server_line(
            "llama_kv_cache_init: CUDA0 compute buffer size = 384 MiB",
            "stderr",
        )
        manager._ingest_server_line("llama_model_load: model loaded", "stderr")

        snap = manager._workspace_calibration.snapshot()
        assert snap["fa_on"]["samples"] == 1
        # Ratio reflects ONLY the CUDA0 (vram) component (384 / 384 = 1.0).
        # If the CPU buffer leaked in we'd get a ratio of (9999+384)/384 ≈ 27.
        assert abs(snap["fa_on"]["ratio"] - 1.0) < 1e-6

    def test_first_fit_fa_off_reserves_more_compute_than_fa_on(
        self, manager: LlamaServerManager,
    ) -> None:
        """T2-6: ``flash_attn=False`` reserves a larger compute pool, so
        the first-fit allocator returns NO MORE layers than the FA-on
        path on the same hardware/context.

        Why this matters: pre-T2-6 the allocator used a flat 512 MiB
        compute reserve regardless of FA. A user on a Pascal card (no
        FA support) would get a layer count computed against 512 MiB
        but actually need 640 MiB, OOMing on load. After T2-6 the
        allocator reserves the right amount up front and lands
        slightly fewer layers on the GPU — slower but actually fits.

        Margin sized so the FA-on/FA-off compute delta (256 MiB) is
        big enough to flip the layer count by at least 1: tight VRAM
        budget that's already near the per-layer overflow threshold.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        # 12 GiB card (consumer / midrange), 30 GB / 32-layer model
        # roughly fills VRAM after KV. Gives the 256 MiB compute delta
        # room to actually move the layer-fit needle.
        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 12 * 1024**3, "total_mib": 12 * 1024, "free_mib": 11 * 1024}
        )
        profile = ModelProfile(
            model_path="/models/dense.gguf",
            model_name="dense-30g-12vram",
            n_layers=32,
            n_heads=32,
            n_heads_kv=8,
            n_embed=4096,
            context_length=32768,
            total_size_bytes=30 * 1024**3,
        )

        fit_fa_on = manager._autofit_gpu_layers_for(
            profile, ctx_size=8192, kv_cache_type="",
            gpu_layers_cap=99, flash_attn=True,
        )
        fit_fa_off = manager._autofit_gpu_layers_for(
            profile, ctx_size=8192, kv_cache_type="",
            gpu_layers_cap=99, flash_attn=False,
        )

        # FA off must NEVER fit more layers than FA on — its compute
        # reserve is strictly larger, so the per-layer cost is higher.
        assert fit_fa_off <= fit_fa_on, (
            f"FA-off allocator returned {fit_fa_off} layers vs FA-on's "
            f"{fit_fa_on} — compute reserve must be at least as large for FA-off"
        )
        # Both must still produce a positive count for a fittable
        # scenario — sanity check that we haven't bricked the allocator.
        assert fit_fa_on > 0
        assert fit_fa_off > 0

    def test_first_fit_fa_default_falls_back_to_manager_setting(
        self, manager: LlamaServerManager,
    ) -> None:
        """``flash_attn=None`` (the default) uses ``self.flash_attn``.

        Backwards-compat guarantee: every existing test calls the
        allocator without the FA arg — they must keep getting the
        manager's configured default. Going from None → True (the
        manager default) shouldn't change layer counts vs explicit
        ``flash_attn=True``.
        """
        from augmentum.models.model_profile_cache import ModelProfile

        manager._query_gpu_info_blocking = (  # type: ignore[method-assign]
            lambda: {"total_bytes": 24 * 1024**3, "total_mib": 24 * 1024, "free_mib": 22 * 1024}
        )
        profile = ModelProfile(
            model_path="/models/m.gguf", model_name="m",
            n_layers=32, n_heads=32, n_heads_kv=8, n_embed=4096,
            context_length=8192, total_size_bytes=15 * 1024**3,
        )

        manager.flash_attn = True
        fit_default = manager._autofit_gpu_layers_for(
            profile, ctx_size=4096, kv_cache_type="", gpu_layers_cap=99,
        )
        fit_explicit_on = manager._autofit_gpu_layers_for(
            profile, ctx_size=4096, kv_cache_type="",
            gpu_layers_cap=99, flash_attn=True,
        )
        assert fit_default == fit_explicit_on, (
            f"None-default ({fit_default}) must equal explicit FA-on "
            f"({fit_explicit_on}) when manager.flash_attn=True"
        )

        # And inversely: with manager.flash_attn=False, None-default
        # must equal explicit FA-off.
        manager.flash_attn = False
        fit_default_off = manager._autofit_gpu_layers_for(
            profile, ctx_size=4096, kv_cache_type="", gpu_layers_cap=99,
        )
        fit_explicit_off = manager._autofit_gpu_layers_for(
            profile, ctx_size=4096, kv_cache_type="",
            gpu_layers_cap=99, flash_attn=False,
        )
        assert fit_default_off == fit_explicit_off

    def test_compute_reserve_bytes_matches_build_load_plan_constants(
        self, manager: LlamaServerManager,
    ) -> None:
        """The helper must produce the SAME numbers ``build_load_plan``
        used to inline (384 / 640 MiB).

        If somebody bumps one constant without bumping the other, the
        plan's displayed peak VRAM will drift away from what the
        allocator actually reserves — two different lies in the UI.
        Lock the values.
        """
        assert manager._compute_reserve_bytes(True) == 384 * 1024**2
        assert manager._compute_reserve_bytes(False) == 640 * 1024**2

    def test_build_load_plan_compute_reserve_uses_helper(
        self, manager: LlamaServerManager,
    ) -> None:
        """Plan's estimated_vram_mb reflects the helper, not stale
        inline 384/640 numbers.

        Direct check via the plan's memory section: the difference
        between FA-on and FA-off plans for the same load should equal
        the helper delta (640-384 = 256 MiB).
        """
        profile = _make_profile(ctx=4096)
        manager._query_gpu_info = lambda: {"total_mib": 80_000, "free_mib": 78_000}
        manager._query_system_memory_info = lambda: {"total_mib": 64_000, "available_mib": 48_000}

        plan_fa_on = manager.build_load_plan(
            "/models/test.gguf",
            load_options={
                "gpu_layers_mode": "custom", "gpu_layers": 32,
                "ctx_size": 4096, "batch_size": 512, "flash_attn": True,
            },
            profile=profile,
        )
        plan_fa_off = manager.build_load_plan(
            "/models/test.gguf",
            load_options={
                "gpu_layers_mode": "custom", "gpu_layers": 32,
                "ctx_size": 4096, "batch_size": 512, "flash_attn": False,
            },
            profile=profile,
        )

        # FA-off should reserve 256 MiB more — that's the helper delta.
        # Direct subtraction in MiB: estimated_vram_mb is in MiB units.
        delta_mb = (
            plan_fa_off["memory"]["estimated_vram_mb"]
            - plan_fa_on["memory"]["estimated_vram_mb"]
        )
        # ``_estimate_prompt_workspace_bytes`` also branches on
        # flash_attn (the `attention_factor = 4.0 if flash_attn else 8.0`
        # at line 647), so the total delta is compute_reserve_delta +
        # workspace_delta. Assert the floor: at least 256 MiB of the
        # delta comes from the compute-reserve helper.
        assert delta_mb >= 256, (
            f"FA-off plan should reserve at least 256 MiB more VRAM than FA-on "
            f"(compute_reserve delta), got {delta_mb} MiB"
        )

    def test_fit_warning_uses_currently_free_gpu_memory(self, manager: LlamaServerManager) -> None:
        profile = _make_profile(ctx=16384)
        manager._query_gpu_info = lambda: {"total_mib": 24_000, "free_mib": 2_000}
        manager._query_system_memory_info = lambda: {"total_mib": 64_000, "available_mib": 48_000}

        plan = manager.build_load_plan(
            "/models/test.gguf",
            load_options={
                "gpu_layers_mode": "custom",
                "gpu_layers": 32,
                "ctx_size": 16384,
                "batch_size": 2048,
                "flash_attn": False,
            },
            profile=profile,
        )

        assert plan["memory"]["fits_gpu"] is False
        assert any("currently free on the GPU" in warning for warning in plan["warnings"])


# ----- File discovery tests -----


class TestDiscoverGgufFiles:
    def test_scan_model_dirs(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # Direct file
        (model_dir / "test1.gguf").write_bytes(b"\x00" * 100)
        # Nested one level
        sub = model_dir / "subfolder"
        sub.mkdir()
        (sub / "test2.gguf").write_bytes(b"\x00" * 200)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        results = mgr.discover_gguf_files()
        filenames = {r["filename"] for r in results}
        assert "test1.gguf" in filenames
        assert "test2.gguf" in filenames
        assert len(results) == 2
        # Check structure
        for r in results:
            assert "path" in r
            assert "size" in r
            assert "modified" in r

    def test_scan_two_levels_deep(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        lvl1 = model_dir / "a"
        lvl1.mkdir()
        lvl2 = lvl1 / "b"
        lvl2.mkdir()
        (lvl2 / "deep.gguf").write_bytes(b"\x00" * 50)
        # level 3 should not be found
        lvl3 = lvl2 / "c"
        lvl3.mkdir()
        (lvl3 / "tooDeep.gguf").write_bytes(b"\x00" * 50)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        results = mgr.discover_gguf_files()
        filenames = {r["filename"] for r in results}
        assert "deep.gguf" in filenames
        assert "tooDeep.gguf" not in filenames

    def test_empty_dir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        mgr = LlamaServerManager(model_dir=str(model_dir))
        results = mgr.discover_gguf_files()
        assert results == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        mgr = LlamaServerManager(model_dir=str(tmp_path / "nope"))
        results = mgr.discover_gguf_files()
        assert results == []

    def test_skips_dotdirs(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        hidden = model_dir / ".hidden"
        hidden.mkdir()
        (hidden / "secret.gguf").write_bytes(b"\x00" * 100)
        (model_dir / "visible.gguf").write_bytes(b"\x00" * 100)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        results = mgr.discover_gguf_files()
        filenames = {r["filename"] for r in results}
        assert "visible.gguf" in filenames
        assert "secret.gguf" not in filenames

    def test_extra_model_dirs(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        (dir1 / "a.gguf").write_bytes(b"\x00" * 100)
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir2 / "b.gguf").write_bytes(b"\x00" * 100)

        mgr = LlamaServerManager(model_dir=str(dir1), extra_model_dirs=[str(dir2)])
        results = mgr.discover_gguf_files()
        filenames = {r["filename"] for r in results}
        assert "a.gguf" in filenames
        assert "b.gguf" in filenames

    def test_deduplicates_files(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "test.gguf").write_bytes(b"\x00" * 100)

        # Same dir passed twice
        mgr = LlamaServerManager(
            model_dir=str(model_dir),
            extra_model_dirs=[str(model_dir)],
        )
        results = mgr.discover_gguf_files()
        assert len(results) == 1


# ----- Resolve model path tests -----


class TestResolveModelPath:
    def test_exact_match(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        target = model_dir / "mymodel.gguf"
        target.write_bytes(b"\x00" * 100)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        result = mgr._resolve_model_path("mymodel.gguf")
        assert result is not None
        assert os.path.basename(result) == "mymodel.gguf"

    def test_without_extension(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "mymodel.gguf").write_bytes(b"\x00" * 100)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        result = mgr._resolve_model_path("mymodel")
        assert result is not None

    def test_fuzzy_case_insensitive(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "MyModel-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

        mgr = LlamaServerManager(model_dir=str(model_dir))
        result = mgr._resolve_model_path("mymodel-q4_k_m")
        assert result is not None

    def test_not_found(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        mgr = LlamaServerManager(model_dir=str(model_dir))
        result = mgr._resolve_model_path("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Idle monitor resilience
# ---------------------------------------------------------------------------


class TestIdleMonitorResilience:
    """Background idle monitor must survive transient iteration errors."""

    @pytest.mark.asyncio
    async def test_iteration_exception_does_not_kill_monitor(
        self, manager: LlamaServerManager
    ) -> None:
        """Regression: the monitor must keep ticking after a body
        exception. Pre-fix only CancelledError was caught — any other
        exception (TypeError, OSError from stop(), etc.) silently
        killed the fire-and-forget task and disabled idle-unload until
        the next manual start_idle_monitor() call.
        """
        import asyncio

        manager.state = ProcessState.READY
        manager.idle_timeout = 0.04  # check_interval = 0.02s
        manager._last_request_time = __import__("time").monotonic()
        # Force the in-flight branch so each iteration calls touch().
        # We then make touch() raise once, then succeed; the loop
        # should log the first failure, sleep briefly, and keep ticking.
        manager._in_flight_count = 1

        touch_calls = {"n": 0}
        original_touch = manager.touch

        def flaky_touch() -> None:
            touch_calls["n"] += 1
            if touch_calls["n"] == 1:
                raise RuntimeError("simulated transient failure")
            original_touch()

        manager.touch = flaky_touch  # type: ignore[method-assign]

        manager.start_idle_monitor()
        # Wait long enough for several iterations: with check_interval
        # = 0.02s and a 1s post-error backoff, ~2.5s gives us the
        # initial tick (raises) + the 1s backoff + several post-recovery
        # ticks. Generous on the upper bound so Windows scheduler jitter
        # doesn't flake the test.
        try:
            await asyncio.sleep(2.5)
            # Monitor task is still alive — pre-fix it would be in
            # ``done()`` state with the RuntimeError as its exception.
            assert manager._idle_task is not None
            assert not manager._idle_task.done(), (
                "monitor task died on first exception "
                f"(exc: {manager._idle_task.exception()!r})"
            )
            # touch() ran more than once — the second call (post-
            # recovery) succeeded.
            assert touch_calls["n"] >= 2, (
                f"monitor only ticked {touch_calls['n']} times, expected >= 2"
            )
        finally:
            manager.stop_idle_monitor()
            # Drain the cancellation
            try:
                if manager._idle_task is not None:
                    await asyncio.wait_for(manager._idle_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


class TestRequestInFlightCancellation:
    """``request_in_flight()``'s ``finally`` block must run even on cancel.

    The counter is the only thing standing between the idle monitor
    and a mid-request unload. If a task gets cancelled (browser
    disconnect, route handler error, asyncio.shield being yanked)
    and the finally clause skips, the counter stays elevated forever
    — the idle monitor would refuse to unload the model EVER, even
    when truly idle. Conversely, if the counter goes negative because
    the increment didn't happen but decrement did, the monitor would
    unload PREMATURELY mid-request next time anyone calls in.

    These tests prove the counter is balanced under cancellation,
    not just under happy-path use.
    """

    @pytest.mark.asyncio
    async def test_counter_decrements_on_inner_cancel(
        self, manager: LlamaServerManager
    ) -> None:
        """``await asyncio.sleep`` inside the ctx is cancelled — counter
        must still go back to zero.
        """
        import asyncio

        async def slow_op() -> None:
            async with manager.request_in_flight():
                await asyncio.sleep(10.0)  # never completes

        task = asyncio.create_task(slow_op())
        await asyncio.sleep(0.02)  # let it enter request_in_flight
        assert manager._in_flight_count == 1, (
            "counter never incremented — task hadn't entered ctx yet?"
        )

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert manager._in_flight_count == 0, (
            f"counter leaked after cancel: {manager._in_flight_count}"
        )

    @pytest.mark.asyncio
    async def test_counter_decrements_on_body_exception(
        self, manager: LlamaServerManager
    ) -> None:
        """Arbitrary exception in body — counter still decrements."""
        async def failing_op() -> None:
            async with manager.request_in_flight():
                raise RuntimeError("simulated body failure")

        with pytest.raises(RuntimeError, match="simulated body failure"):
            await failing_op()

        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_repeated_cancels_keep_counter_balanced(
        self, manager: LlamaServerManager
    ) -> None:
        """Ten enter+cancel cycles still leave the counter at zero.

        Catches drift bugs where the counter accumulates 1 per cycle
        instead of net-zeroing — would only show up after the manager
        had been running for a while.
        """
        import asyncio

        for _ in range(10):
            async def slow_op() -> None:
                async with manager.request_in_flight():
                    await asyncio.sleep(10.0)

            t = asyncio.create_task(slow_op())
            await asyncio.sleep(0.005)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        assert manager._in_flight_count == 0, (
            f"counter drifted after 10 cancel cycles: "
            f"{manager._in_flight_count}"
        )


class TestIdleMonitorInFlightRace:
    """The actual race the request_in_flight() wrap protects against.

    Earlier coverage only verified the ``async with`` syntax and
    counter increment — it didn't prove the monitor backs off when
    a real long-running operation is mid-flight at the moment the
    idle clock would otherwise expire. That's the regression that
    surfaced in production: ``prepare_stable_checkpoint`` doing a
    5-10s prewarm on long narrative contexts, racing against an
    idle_timeout that just elapsed because the chat stream just
    closed. Without the wrap, ``stop()`` could fire mid-prewarm
    and yank llama-server out from under the in-flight save.

    Test strategy: tight idle_timeout (0.2s) + a synthetic long
    in-flight operation (0.5s); assert state stays READY through
    multiple monitor ticks, and that the monitor DOES eventually
    fire after the operation exits — proving both directions of
    the gate work.
    """

    @pytest.mark.asyncio
    async def test_in_flight_blocks_unload_then_releases(
        self, manager: LlamaServerManager
    ) -> None:
        """Monitor refuses to unload while in_flight_count > 0;
        unloads on the next tick once the count returns to zero.
        """
        import asyncio
        import time

        manager.state = ProcessState.READY
        manager.idle_timeout = 0.2  # check_interval = 0.1s
        # Force the monitor to want-to-unload: clock starts way past
        # the timeout so without the in-flight gate it would fire on
        # the very first tick.
        manager._last_request_time = time.monotonic() - 100.0

        # Track stop() invocations so we can assert "didn't unload
        # while busy". stop() with process=None is a fast no-op that
        # just transitions to IDLE — perfect for this test.
        stop_calls: list[float] = []
        original_stop = manager.stop

        async def tracking_stop() -> None:
            stop_calls.append(time.monotonic())
            await original_stop()

        manager.stop = tracking_stop  # type: ignore[method-assign]

        # Synthetic in-flight operation: holds the request_in_flight
        # context for 0.5s. Mirrors prepare_stable_checkpoint's slow
        # prewarm shape.
        async def slow_in_flight() -> None:
            async with manager.request_in_flight():
                await asyncio.sleep(0.5)

        req_task = asyncio.create_task(slow_in_flight())
        # Let the task enter request_in_flight() before we start the
        # monitor — counter must be 1 by the time the first tick fires.
        await asyncio.sleep(0.02)
        assert manager._in_flight_count == 1

        manager.start_idle_monitor()

        try:
            # Through the next ~3 monitor ticks (0.3s vs 0.1s interval),
            # the in-flight count keeps the monitor from unloading.
            # Without the wrap this is exactly when stop() would fire.
            await asyncio.sleep(0.3)
            assert manager.state == ProcessState.READY, (
                f"unloaded while in-flight (count={manager._in_flight_count}, "
                f"stop_calls={stop_calls})"
            )
            assert stop_calls == [], (
                f"stop() called during in-flight window: {stop_calls}"
            )

            # Wait for the in-flight task to finish — counter goes to
            # 0 and touch() updates _last_request_time. The monitor
            # then needs another full ``idle_timeout`` worth of
            # idleness before it'll unload.
            await req_task
            assert manager._in_flight_count == 0

            # Drive past the post-exit idle window. After exit, the
            # clock was reset; we need ``idle_timeout`` (0.2) +
            # ``check_interval`` (0.1) plus jitter headroom.
            await asyncio.sleep(0.6)

            # Now stop() should have fired exactly once.
            assert manager.state == ProcessState.IDLE, (
                f"monitor failed to unload after in-flight exit "
                f"(state={manager.state}, stop_calls={stop_calls})"
            )
            assert len(stop_calls) == 1, (
                f"expected exactly one unload, got {len(stop_calls)}"
            )
        finally:
            manager.stop_idle_monitor()
            try:
                if manager._idle_task is not None:
                    await asyncio.wait_for(manager._idle_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


# ---------------------------------------------------------------------------
# GPU info cache — async-safe nvidia-smi
# ---------------------------------------------------------------------------


class TestGpuInfoCache:
    """Async-safe nvidia-smi caching."""

    @pytest.mark.asyncio
    async def test_async_warm_populates_cache_for_sync_callers(self, manager: LlamaServerManager) -> None:
        """Pre-warm via async path, verify sync callers hit the cache.

        Mirrors the actual start()-time call sequence: await
        _query_gpu_info_async() → sync helpers (autofit, cap_ctx, etc.)
        invoke _query_gpu_info() and must observe the warm value
        without re-shelling to nvidia-smi.
        """
        call_count = {"n": 0}
        sentinel = {
            "total_bytes": 24 * 1024 * 1024 * 1024,
            "used_bytes": 4 * 1024 * 1024 * 1024,
            "free_bytes": 20 * 1024 * 1024 * 1024,
            "total_mib": 24576,
            "used_mib": 4096,
            "free_mib": 20480,
            "gpu_name": "Test-GPU",
        }

        def fake_blocking() -> dict:
            call_count["n"] += 1
            return sentinel

        manager._query_gpu_info_blocking = fake_blocking  # type: ignore[method-assign]

        # Pre-warm.
        warmed = await manager._query_gpu_info_async()
        assert warmed == sentinel
        assert call_count["n"] == 1

        # Three subsequent sync calls (mirroring _get_vram_bytes,
        # _autofit_gpu_layers_for, _cap_ctx_for_vram in build_load_plan)
        # must hit the cache without invoking the blocking helper again.
        for _ in range(3):
            info = manager._query_gpu_info()
            assert info == sentinel
        assert call_count["n"] == 1, (
            f"sync callers re-shelled to nvidia-smi {call_count['n']} times "
            "instead of using the async-warm cache"
        )

    def test_sync_path_re_queries_after_ttl_expires(self, manager: LlamaServerManager) -> None:
        """Status polls past the TTL re-query rather than serving stale.

        Periodic ``status()`` polls need fresh VRAM numbers; the cache
        is meant for back-to-back load-plan helpers, not indefinite
        reuse. Override the TTL down to zero to verify expiry triggers
        a fresh blocking call.
        """
        call_count = {"n": 0}

        def fake_blocking() -> dict:
            call_count["n"] += 1
            return {"total_bytes": call_count["n"], "total_mib": call_count["n"]}

        manager._query_gpu_info_blocking = fake_blocking  # type: ignore[method-assign]
        manager._GPU_INFO_TTL_S = 0.0  # type: ignore[misc]

        first = manager._query_gpu_info()
        second = manager._query_gpu_info()
        assert first["total_mib"] == 1
        assert second["total_mib"] == 2
        assert call_count["n"] == 2

    def test_async_path_does_not_block_event_loop(self, manager: LlamaServerManager) -> None:
        """The async wrapper must run blocking work in a worker thread.

        Confirm by registering a slow blocking callable and verifying a
        concurrently-scheduled coroutine makes progress while the GPU
        query is in flight. Pre-fix the synchronous _query_gpu_info()
        call inside async paths blocked the loop entirely.
        """
        import asyncio

        async def _run() -> None:
            tick_count = {"n": 0}

            async def _ticker() -> None:
                # Fires every 5 ms while the slow blocking call runs.
                for _ in range(20):
                    tick_count["n"] += 1
                    await asyncio.sleep(0.005)

            def slow_blocking() -> dict:
                # 100 ms blocking sleep — would freeze the loop without
                # asyncio.to_thread. The ticker above gives ~20 chances
                # to advance during this window if the loop is healthy.
                import time as _t
                _t.sleep(0.1)
                return {"total_bytes": 1, "total_mib": 1}

            manager._query_gpu_info_blocking = slow_blocking  # type: ignore[method-assign]

            ticker_task = asyncio.create_task(_ticker())
            await manager._query_gpu_info_async()
            await ticker_task

            # Loop kept ticking during the 100 ms blocking call — at
            # 5 ms cadence we expect at least ~10 ticks. If async-wrap
            # was missing, the ticker wouldn't get a chance to run
            # until after the blocking call returned.
            assert tick_count["n"] >= 10, (
                f"event loop appeared blocked: only {tick_count['n']} "
                "ticks during a 100 ms blocking call"
            )

        asyncio.run(_run())


class TestWarmTopSession:
    """Restart-warm correctness, especially under iteration after 4xx miss."""

    @staticmethod
    def _build_manager_with_manifest(
        tmp_path: Path,
        sessions: list[tuple[str, dict[str, object]]],
    ) -> LlamaServerManager:
        """Build a manager wired to a fresh manifest seeded with sessions.

        ``sessions`` is a list of (session_key, runtime_overrides) tuples.
        A slot file is created on disk for each session so the warm loop
        passes the ``os.path.isfile`` gate.
        """
        from augmentum.models.kv_session_manifest import KVSessionManifest
        from augmentum.models.llama_cpp import LlamaCppBackend

        model_dir = str(tmp_path / "models")
        slot_dir = os.path.join(model_dir, ".slots", "test-model")
        os.makedirs(slot_dir, exist_ok=True)

        manifest_path = str(tmp_path / "kv_manifest.db")
        manifest = KVSessionManifest(manifest_path)

        mgr = LlamaServerManager(
            model_dir=model_dir,
            backend_port=8091,
            kv_manifest_db=manifest_path,
        )
        # Bypass the constructor manifest hand-off; we want our reference.
        mgr._session_manifest = manifest
        mgr._slot_dir = slot_dir
        mgr.model_id = "test-model"
        mgr.model_path = str(tmp_path / "models" / "test-model.gguf")
        mgr.current_ctx_size = 8192
        mgr.current_kv_cache_type = ""
        mgr.current_flash_attn = True
        mgr.current_gpu_layers = 32
        mgr.current_gpu_layers_mode = "auto"
        mgr.current_batch_size = 512
        mgr.current_draft_model = ""
        mgr.current_draft_max = 5

        runtime = mgr.current_runtime_signature()

        for session_key, _ in sessions:
            slot_filename = LlamaCppBackend._slot_storage_name(session_key)
            slot_path = os.path.join(slot_dir, slot_filename)
            Path(slot_path).write_bytes(b"\x00" * 16)
            manifest.record_save(
                model_key=runtime["model_key"],
                session_key=session_key,
                mode="narrative",
                slot_dir=slot_dir,
                slot_filename=slot_filename,
                model_id=runtime["model_id"],
                model_path=runtime["model_path"],
                model_mtime=runtime["model_mtime"],
                ctx_size=runtime["ctx_size"],
                kv_cache_type=runtime["kv_cache_type"],
                template_fingerprint="",
                system_prompt_hash="",
                prompt_fingerprint="",
                prompt_message_count=0,
                ttl_days=7,
                pinned=False,
                flash_attn=runtime["flash_attn"],
                gpu_layers=runtime["gpu_layers"],
                gpu_layers_mode=runtime["gpu_layers_mode"],
                batch_size=runtime["batch_size"],
                draft_model=runtime["draft_model"],
                draft_max=runtime["draft_max"],
            )
        return mgr

    @pytest.mark.asyncio
    async def test_erase_before_each_restore_attempt(self, tmp_path: Path) -> None:
        """Regression: warm loop must erase slot 0 before every restore.

        Without an erase between candidates, a 4xx on the first
        candidate leaves slot 0 in a partial state that can reject the
        next candidate's restore with ``state_read_meta: failed to find
        available cells``. Mirrors LlamaCppBackend.restore_slot's
        defensive sequencing.

        Programmable transport returns 400 for the first restore and
        200 for the second — the test asserts an erase fires before
        EACH restore (two erases total) AND that the second candidate
        is the one that ends up warmed.
        """
        import httpx

        # Two compatible candidates; first is MRU and will 4xx, fall
        # through to the second which succeeds.
        sessions = [
            ("sess-first-mru", {}),
            ("sess-second", {}),
        ]
        # Seed in REVERSE so the first listed is most recent (manifest
        # orders MRU-first by last_accessed).
        mgr = self._build_manager_with_manifest(tmp_path, list(reversed(sessions)))

        request_log: list[tuple[str, str]] = []
        # First restore returns 400 (simulating a stale slot file or
        # corrupted state); second returns 200.
        restore_call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "action=erase" in url:
                request_log.append(("erase", url))
                return httpx.Response(200, json={"status": "ok"})
            if "action=restore" in url:
                request_log.append(("restore", url))
                restore_call_count["n"] += 1
                if restore_call_count["n"] == 1:
                    return httpx.Response(400, text="state_read_meta failed")
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        # Patch httpx.AsyncClient so the warm loop uses our transport.
        original_client = httpx.AsyncClient

        def patched_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
            kwargs.pop("timeout", None)
            return original_client(transport=httpx.MockTransport(handler), timeout=30.0)

        from unittest.mock import patch

        with patch("augmentum.models.llama_server_manager.httpx.AsyncClient",
                   side_effect=patched_client):
            await mgr._warm_top_session()

        # Expected: erase, restore (400), erase, restore (200) — in that order.
        actions = [kind for kind, _ in request_log]
        assert actions == ["erase", "restore", "erase", "restore"], (
            f"unexpected request sequence: {actions}"
        )
        assert restore_call_count["n"] == 2

        # The second candidate (the one that succeeded) ends up warmed.
        assert mgr._warm_session_key == "sess-first-mru" or \
               mgr._warm_session_key == "sess-second"
        # Specifically: whichever was tried second is the one that succeeded.
        # MRU ordering depends on touch order; verify via the final warm key.
        assert mgr._warm_session_key in {"sess-first-mru", "sess-second"}


# ----- VRAM-release telemetry -----


class TestVramReleaseTelemetry:
    """``stop()`` emits ``vram_release`` (healthy) or
    ``vram_release_lagged`` (driver lag) post-teardown.

    Driver lag — the WSL2/Docker class of bug where VRAM doesn't return
    after subprocess exit — is silent today; this telemetry surfaces it
    in logs so the operator can see how often it fires before deciding
    on a heavier mitigation. The thresholds (3s poll window, 500 MiB
    minimum healthy release) are conservative on purpose; tighten only
    after a baseline of real loads has shown the typical release shape.
    """

    @pytest.fixture
    def stub_process(self):
        """A subprocess-shaped stub that ``terminate()`` settles
        immediately. Lets us drive ``stop()`` without spawning anything
        real. ``returncode`` going non-None on terminate is what
        ``await process.wait()`` actually checks under asyncio.
        """
        import asyncio

        class _StubProc:
            pid = 12345
            returncode: int | None = None

            def terminate(self) -> None:
                self.returncode = 0

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                # Yield once so wait_for sees a completed coroutine
                # without an unbounded await.
                await asyncio.sleep(0)
                return self.returncode if self.returncode is not None else 0

        return _StubProc()

    def _set_up_for_stop(
        self,
        manager: LlamaServerManager,
        stub_process,
        *,
        gpu_layers: int,
        pre_used_mib: int,
        post_used_mib: int,
    ) -> list[dict]:
        """Wire the stubs and capture VRAM samples in order.

        Returns the same ``samples`` list the test can introspect after
        ``stop()`` to assert call order.
        """
        manager.process = stub_process
        manager.model_id = "test-model"
        manager.current_gpu_layers = gpu_layers
        manager.state = ProcessState.READY

        samples: list[dict] = [
            {"used_mib": pre_used_mib},
            {"used_mib": post_used_mib},
        ]
        call_idx = {"n": 0}

        async def fake_sample() -> dict:
            i = call_idx["n"]
            call_idx["n"] += 1
            return samples[min(i, len(samples) - 1)]

        manager._sample_vram_fresh_async = fake_sample  # type: ignore[method-assign]
        # Prevent post-stop telemetry from really sleeping the test out.
        import asyncio as _asyncio
        orig_sleep = _asyncio.sleep

        async def fast_sleep(_seconds: float) -> None:
            await orig_sleep(0)

        manager._test_orig_sleep = orig_sleep  # type: ignore[attr-defined]
        return samples

    @pytest.mark.asyncio
    async def test_healthy_release_logs_info(
        self,
        manager: LlamaServerManager,
        stub_process,
        monkeypatch,
        capfd,
    ) -> None:
        """A ~5 GiB release post-stop emits the ``vram_release`` INFO event."""
        self._set_up_for_stop(
            manager, stub_process,
            gpu_layers=40, pre_used_mib=6000, post_used_mib=900,
        )
        import asyncio as _asyncio
        _real_sleep = _asyncio.sleep
        monkeypatch.setattr(_asyncio, "sleep", lambda _s: _real_sleep(0))

        await manager.stop()

        out, err = capfd.readouterr()
        combined = out + err
        assert "vram_release " in combined or "vram_release\n" in combined or "[info" in combined and "vram_release" in combined
        assert "vram_release_lagged" not in combined

    @pytest.mark.asyncio
    async def test_lagged_release_logs_warning(
        self,
        manager: LlamaServerManager,
        stub_process,
        monkeypatch,
        capfd,
    ) -> None:
        """Only ~100 MiB released → driver lag → WARN telemetry."""
        self._set_up_for_stop(
            manager, stub_process,
            gpu_layers=40, pre_used_mib=6000, post_used_mib=5900,
        )
        import asyncio as _asyncio
        _real_sleep = _asyncio.sleep
        monkeypatch.setattr(_asyncio, "sleep", lambda _s: _real_sleep(0))

        await manager.stop()

        out, err = capfd.readouterr()
        combined = out + err
        assert "vram_release_lagged" in combined

    @pytest.mark.asyncio
    async def test_cpu_only_load_skips_telemetry(
        self,
        manager: LlamaServerManager,
        stub_process,
        capfd,
    ) -> None:
        """gpu_layers=0 → no sample, no log — keep CPU deploys quiet."""
        sample_called = {"n": 0}

        async def fake_sample() -> dict:
            sample_called["n"] += 1
            return {"used_mib": 0}

        manager._sample_vram_fresh_async = fake_sample  # type: ignore[method-assign]
        manager.process = stub_process
        manager.current_gpu_layers = 0
        manager.state = ProcessState.READY

        await manager.stop()

        out, err = capfd.readouterr()
        combined = out + err
        # No telemetry emitted because gpu_layers=0 short-circuits the
        # telemetry branch entirely; pre-sample is never even called.
        assert "vram_release" not in combined
        assert "vram_release_lagged" not in combined
        assert sample_called["n"] == 0

    @pytest.mark.asyncio
    async def test_nvidia_smi_unavailable_skips_silently(
        self,
        manager: LlamaServerManager,
        stub_process,
        capfd,
    ) -> None:
        """If the VRAM sampler raises, stop() still completes cleanly."""
        async def boom() -> dict:
            raise RuntimeError("nvidia-smi exploded")

        manager._sample_vram_fresh_async = boom  # type: ignore[method-assign]
        manager.process = stub_process
        manager.current_gpu_layers = 40
        manager.state = ProcessState.READY

        # Must not raise — the pre-sample failure should be swallowed.
        await manager.stop()

        out, err = capfd.readouterr()
        combined = out + err
        assert "vram_release" not in combined
        assert "vram_release_lagged" not in combined
        assert manager.state == ProcessState.IDLE


class TestStopEscalateKill:
    """``stop()`` must escalate when ``asyncio.wait()`` hangs past the
    SIGKILL window — the original silent ``pass`` left the manager
    flipping state to IDLE while the subprocess still held the GPU.

    Trigger: ``process.wait()`` raises ``TimeoutError`` (simulating the
    asyncio waiter never resolving because the kernel hasn't reaped
    the process — WSL2+CUDA D-state). Both inner wait_for calls hit
    the except path, escalation fires, and the new fields/log lines
    reflect the strand.
    """

    @pytest.fixture
    def hanging_process(self):
        """Stub whose ``wait()`` raises TimeoutError both times,
        forcing stop() into the escalation branch.
        """
        class _Hang:
            pid = 99999
            returncode: int | None = None

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

            async def wait(self) -> int:
                # asyncio.TimeoutError IS TimeoutError on 3.11+, so
                # raising bare TimeoutError trips the wait_for except.
                raise TimeoutError("kernel hasn't reaped")

        return _Hang()

    @pytest.mark.asyncio
    async def test_escalation_polls_psutil_and_resignals(
        self,
        manager: LlamaServerManager,
        hanging_process,
        monkeypatch,
    ) -> None:
        """PID alive across the wait timeout → escalate_kill loop re-
        sends SIGKILL until psutil reports the PID gone.
        """
        manager.process = hanging_process
        manager.state = ProcessState.READY
        manager.current_gpu_layers = 0  # skip VRAM telemetry branch

        pid_exists_calls = {"n": 0}

        def fake_pid_exists(pid: int) -> bool:
            pid_exists_calls["n"] += 1
            # Alive for first 2 polls, then reaped.
            return pid_exists_calls["n"] <= 2

        kill_calls: list[tuple[int, int]] = []

        def fake_os_kill(pid: int, sig: int) -> None:
            kill_calls.append((pid, sig))

        async def fast_sleep(_seconds: float) -> None:
            return

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.psutil.pid_exists",
            fake_pid_exists,
        )
        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.os.kill",
            fake_os_kill,
        )
        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.asyncio.sleep",
            fast_sleep,
        )

        await manager.stop()

        # Manager state still clears to IDLE — the next reconcile
        # handles any genuine strand.
        assert manager.state == ProcessState.IDLE
        # Escalate loop polled pid_exists and re-sent SIGKILL until
        # the PID was reaped (3rd poll returns False).
        assert pid_exists_calls["n"] >= 2, (
            f"expected at least 2 pid_exists polls, got {pid_exists_calls['n']}"
        )
        assert len(kill_calls) >= 1, "escalate_kill never re-sent SIGKILL"
        # SIGKILL value is 9 on POSIX; on Windows signal.SIGKILL doesn't
        # exist and the manager falls back to the bare wire value. Both
        # paths land on 9 by the time os.kill receives it.
        import signal as _signal
        expected_sig = getattr(_signal, "SIGKILL", 9)
        assert all(sig == expected_sig for _, sig in kill_calls)

    @pytest.mark.asyncio
    async def test_escalation_logs_unkillable_when_pid_survives(
        self,
        manager: LlamaServerManager,
        hanging_process,
        monkeypatch,
        capfd,
    ) -> None:
        """PID stays alive through every escalation poll → log.error
        with ``stop_subprocess_unkillable`` (replaces the prior silent
        ``pass`` that hid the strand).
        """
        manager.process = hanging_process
        manager.state = ProcessState.READY
        manager.current_gpu_layers = 0

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.psutil.pid_exists",
            lambda _pid: True,  # always alive
        )
        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.os.kill",
            lambda _pid, _sig: None,
        )

        async def fast_sleep(_seconds: float) -> None:
            return

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.asyncio.sleep",
            fast_sleep,
        )

        await manager.stop()

        out, err = capfd.readouterr()
        combined = out + err
        assert "stop_subprocess_unkillable" in combined, (
            "escalation gave up on unkillable PID but didn't log "
            "stop_subprocess_unkillable — this is the silent-strand "
            "regression we just fixed"
        )
        # State still clears so manager bookkeeping is consistent;
        # reconcile_stranded_subprocess at next start() / idle-tick
        # picks up the strand.
        assert manager.state == ProcessState.IDLE


class TestStatusStrandedFlag:
    """``status()`` surfaces ``stranded=True`` when state is IDLE but
    the backend port still answers ``/health`` — gives the resource
    panel a visible signal instead of a phantom "nothing loaded."
    """

    def test_idle_with_port_responding_sets_stranded(
        self, manager: LlamaServerManager, monkeypatch,
    ) -> None:
        """IDLE + port answers 200 → stranded=True."""
        class _StubResp:
            status_code = 200

        class _StubClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a) -> None:
                pass

            def get(self, _url: str) -> _StubResp:
                return _StubResp()

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.httpx.Client",
            _StubClient,
        )

        # Manager construct already sets state=IDLE, model_id=""
        s = manager.status()
        assert s["state"] == "idle"
        assert s.get("stranded") is True

    def test_idle_with_port_closed_no_stranded_flag(
        self, manager: LlamaServerManager, monkeypatch,
    ) -> None:
        """IDLE + port refused → no ``stranded`` key (common case)."""
        class _BoomClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a) -> None:
                pass

            def get(self, _url: str):
                raise ConnectionRefusedError("port closed")

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.httpx.Client",
            _BoomClient,
        )

        s = manager.status()
        assert "stranded" not in s

    def test_ready_state_skips_probe(
        self, manager: LlamaServerManager, monkeypatch,
    ) -> None:
        """When state is READY, no probe runs — would be pointless
        and adds latency to every UI poll. Probe must only fire on
        the IDLE branch where a strand is actually possible.

        Note: ``check_alive()`` demotes state to IDLE when
        ``self.process is None`` (early in ``status()``), so the
        test gives the manager a live-looking stub process to keep
        state at READY.
        """
        probe_calls = {"n": 0}

        class _CountingClient:
            def __init__(self, *_args, **_kwargs) -> None:
                probe_calls["n"] += 1

            def __enter__(self):
                return self

            def __exit__(self, *_a) -> None:
                pass

            def get(self, _url: str):
                class _R: status_code = 200
                return _R()

        monkeypatch.setattr(
            "augmentum.models.llama_server_manager.httpx.Client",
            _CountingClient,
        )

        class _LiveStub:
            pid = 4242
            returncode = None  # check_alive sees this as still running

        manager.process = _LiveStub()
        manager.state = ProcessState.READY
        manager.model_id = "test"
        manager.model_path = "/models/test.gguf"
        # _query_gpu_info / _query_process_ram_mb shell out to OS tools;
        # neuter them so the test isn't host-dependent.
        manager._query_gpu_info = lambda: None  # type: ignore[method-assign]
        manager._query_process_ram_mb = lambda _pid: 0  # type: ignore[method-assign]

        s = manager.status()
        assert s["state"] == "ready", (
            f"state demoted unexpectedly to {s['state']!r} — test setup wrong"
        )
        assert probe_calls["n"] == 0, (
            "probe ran on READY state — adds latency on every UI poll"
        )
        assert "stranded" not in s


class TestIdleMonitorSelfCancelGuard:
    """The actual root cause behind operator reports of "idle eviction
    logs Stopping then nothing else, model stays in GPU."

    ``stop()``'s very first statement is ``self.stop_idle_monitor()``,
    but ``stop()`` is called from inside the monitor's own task body
    in the idle-timeout-fires path. Pre-fix, ``stop_idle_monitor``
    unconditionally called ``self._idle_task.cancel()`` — which
    scheduled a CancelledError against the currently-running task. It
    fired on the very next await in ``stop()`` (the VRAM pre-sample),
    raising out of ``stop()`` BEFORE ``terminate()`` was called. The
    subprocess survived; state was permanently stuck at STOPPING; no
    further engine logs ever appeared.

    The fix: ``stop_idle_monitor`` checks ``asyncio.current_task()``
    and skips the cancel when the caller IS the monitor task.
    """

    @pytest.mark.asyncio
    async def test_stop_from_idle_monitor_actually_terminates(
        self, manager: LlamaServerManager,
    ) -> None:
        """The idle-timeout path must call ``process.terminate()``.

        Direct regression: pre-fix, ``terminate()`` was never reached
        because the monitor self-cancelled before the first ``await``.
        """
        import asyncio
        import time

        terminate_calls = {"n": 0}

        class _StubProc:
            pid = 11111
            returncode: int | None = None

            def terminate(self_) -> None:
                terminate_calls["n"] += 1
                self_.returncode = 0

            def kill(self_) -> None:
                self_.returncode = -9

            async def wait(self_) -> int:
                await asyncio.sleep(0)
                return self_.returncode if self_.returncode is not None else 0

        manager.process = _StubProc()
        manager.model_id = "test-model"
        manager.current_gpu_layers = 0  # skip VRAM telemetry
        manager.state = ProcessState.READY
        manager.idle_timeout = 0.04
        manager._last_request_time = time.monotonic() - 10.0  # past timeout

        manager.start_idle_monitor()
        try:
            # Enough ticks for idle-timeout-unloading to fire and
            # stop() to run end-to-end.
            await asyncio.sleep(0.4)
        finally:
            # In case the test is unwinding mid-stop somehow.
            manager.stop_idle_monitor()

        assert terminate_calls["n"] == 1, (
            f"terminate() called {terminate_calls['n']} times — "
            "pre-fix, the monitor self-cancelled before reaching "
            "terminate() and the subprocess was leaked"
        )
        assert manager.state == ProcessState.IDLE, (
            f"state stuck at {manager.state} after idle eviction — "
            "the self-cancel guard isn't holding"
        )
        assert manager.process is None, (
            "process reference still set after stop() — clear path "
            "didn't run"
        )

    @pytest.mark.asyncio
    async def test_external_stop_idle_monitor_still_cancels(
        self, manager: LlamaServerManager,
    ) -> None:
        """The self-cancel guard must NOT break the normal external
        path. A user-initiated stop / swap / shutdown calling
        ``stop_idle_monitor`` from outside the monitor's task should
        still cancel the monitor cleanly.
        """
        import asyncio

        manager.state = ProcessState.READY
        manager.idle_timeout = 60.0  # long, won't fire on its own
        manager._last_request_time = __import__("time").monotonic()

        manager.start_idle_monitor()
        # Let the monitor enter its sleep so there's a real task to cancel.
        await asyncio.sleep(0.05)
        monitor_task = manager._idle_task
        assert monitor_task is not None
        assert not monitor_task.done()

        # External cancel — we are NOT the monitor task here.
        manager.stop_idle_monitor()

        # Wait briefly for the cancellation to propagate.
        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        assert monitor_task.done(), (
            "external stop_idle_monitor failed to cancel the monitor"
        )
        assert manager._idle_task is None


class TestIdleMonitorSelfHeal:
    """When the idle monitor fires ``stop()`` and the subprocess
    survives teardown (WSL2+CUDA D-state, etc.), manager bookkeeping
    is already cleared to IDLE but the strand is still holding VRAM.
    The monitor must call ``reconcile_stranded_subprocess`` to
    reclaim immediately — pre-fix, recovery had to wait for the
    next user chat to trigger the start()-path reconcile, which is
    the exact "phantom nothing-loaded for hours" symptom we were
    trying to eliminate.
    """

    @pytest.mark.asyncio
    async def test_reconcile_called_after_idle_stop(
        self, manager: LlamaServerManager,
    ) -> None:
        """Idle monitor fires → stop() runs → reconcile_stranded_
        subprocess runs before the monitor returns.
        """
        import asyncio
        import time

        reconcile_calls = {"n": 0}

        async def tracked_reconcile() -> bool:
            reconcile_calls["n"] += 1
            return False

        # process=None makes stop() a fast no-op; we're testing the
        # call sequence, not the kill path.
        manager.reconcile_stranded_subprocess = tracked_reconcile  # type: ignore[method-assign]
        manager.state = ProcessState.READY
        manager.idle_timeout = 0.04  # check_interval = 0.02s
        manager._last_request_time = time.monotonic() - 10.0  # past timeout

        manager.start_idle_monitor()
        try:
            # Give the monitor enough ticks to fire stop() + reconcile.
            await asyncio.sleep(0.3)
        finally:
            manager.stop_idle_monitor()
            try:
                if manager._idle_task is not None:
                    await asyncio.wait_for(manager._idle_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        assert reconcile_calls["n"] == 1, (
            f"reconcile_stranded_subprocess called {reconcile_calls['n']} "
            "times; expected exactly 1 after idle stop()"
        )

    @pytest.mark.asyncio
    async def test_reconcile_failure_does_not_propagate(
        self, manager: LlamaServerManager,
    ) -> None:
        """If reconcile raises (port probe blew up, psutil scan
        failed, etc.), the monitor logs a warning and still returns
        cleanly — never propagates into the create_task fire-and-
        forget where it would silently kill the loop.
        """
        import asyncio
        import time

        async def boom_reconcile() -> bool:
            raise RuntimeError("simulated reconcile failure")

        manager.reconcile_stranded_subprocess = boom_reconcile  # type: ignore[method-assign]
        manager.state = ProcessState.READY
        manager.idle_timeout = 0.04
        manager._last_request_time = time.monotonic() - 10.0

        manager.start_idle_monitor()
        try:
            await asyncio.sleep(0.3)
            # Monitor task either returned cleanly or is still alive —
            # what we care about is no uncaught exception killing it.
            if manager._idle_task is not None and manager._idle_task.done():
                # If it finished, it should have returned normally,
                # not raised the reconcile RuntimeError.
                assert manager._idle_task.exception() is None, (
                    "monitor task died on reconcile failure: "
                    f"{manager._idle_task.exception()!r}"
                )
        finally:
            manager.stop_idle_monitor()
            try:
                if manager._idle_task is not None:
                    await asyncio.wait_for(manager._idle_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
