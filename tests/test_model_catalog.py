from augmentum.models.model_catalog import list_catalog, resolve_model_name


def test_resolve_current_recommendation_aliases():
    assert resolve_model_name("qwen3.6-27b") == (
        "unsloth/Qwen3.6-27B-GGUF",
        "Qwen3.6-27B-UD-Q4_K_XL.gguf",
    )
    assert resolve_model_name("phi-4") == (
        "unsloth/phi-4-GGUF",
        "phi-4-Q4_K_M.gguf",
    )
    assert resolve_model_name("glm-4.7-flash") == (
        "unsloth/GLM-4.7-Flash-GGUF",
        "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
    )


def test_direct_hugging_face_references_passthrough():
    assert resolve_model_name("org/repo:model.gguf") == ("org/repo", "model.gguf")


def test_explicit_compatibility_quants_still_resolve():
    assert resolve_model_name("qwen3.6-27b:q4_k_m") == (
        "unsloth/Qwen3.6-27B-GGUF",
        "Qwen3.6-27B-Q4_K_M.gguf",
    )
    assert resolve_model_name("qwen3.6-35b-a3b:q4_k_m") == (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )


def test_stale_qwen35_32b_alias_removed_from_catalog():
    assert resolve_model_name("qwen3.5-32b") is None
    names = {entry["name"] for entry in list_catalog()}
    assert "qwen3.5-32b" not in names


def test_catalog_surfaces_new_recommendations_first():
    names = [entry["name"] for entry in list_catalog()[:6]]
    assert names == [
        "qwen3.6-27b",
        "qwen3.6-35b-a3b",
        "gemma-4-e4b-it",
        "magistral-small-2507",
        "phi-4",
        "glm-4.7-flash",
    ]
