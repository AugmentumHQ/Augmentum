"""Tests for the per-tool permission policy resolver."""

from __future__ import annotations

from augmentum.coder.policy import (
    Policy,
    _Rule,
    builtin_default_policy,
    parse_policy_text,
    policy_as_dict,
    policy_to_toml,
)


# ---------------------------------------------------------------------------
# Builtin default policy
# ---------------------------------------------------------------------------

class TestBuiltinDefaults:
    def test_read_only_tools_allow_by_default(self):
        policy = builtin_default_policy()
        assert policy.decide("file_read", {"path": "/x"}) == "allow"
        assert policy.decide("code_grep", {"pattern": "foo"}) == "allow"
        assert policy.decide("find_files", {"pattern": "*.py"}) == "allow"

    def test_write_tools_ask_by_default(self):
        policy = builtin_default_policy()
        assert policy.decide("file_write", {"path": "/x"}) == "ask"
        assert policy.decide("code_edit", {"path": "/x"}) == "ask"
        assert policy.decide("shell_exec", {"command": "ls"}) == "ask"

    def test_unknown_tool_falls_back_to_ask(self):
        policy = builtin_default_policy()
        assert policy.decide("totally_made_up_tool", {}) == "ask"

    def test_http_request_always_asks_without_rule(self):
        # http_request is on the "always ask" floor — no auto-allow
        # by builtin defaults.
        policy = builtin_default_policy()
        assert policy.decide("http_request", {"url": "https://x"}) == "ask"


# ---------------------------------------------------------------------------
# Policy parsing
# ---------------------------------------------------------------------------

class TestPolicyParsing:
    def test_simple_rule_compiles(self):
        text = """
        [[rule]]
        tool = "shell_exec"
        action = "allow"
        """
        policy = parse_policy_text(text)
        assert len(policy.rules) == 1
        assert policy.rules[0].tool == "shell_exec"
        assert policy.rules[0].action == "allow"

    def test_arg_glob_compiles(self):
        text = """
        [[rule]]
        tool = "shell_exec"
        arg_glob = { command = "git *" }
        action = "allow"
        """
        policy = parse_policy_text(text)
        assert policy.decide("shell_exec", {"command": "git status"}) == "allow"
        # Non-matching args fall through to fallback (ask).
        assert policy.decide("shell_exec", {"command": "rm -rf /"}) == "ask"

    def test_first_match_wins(self):
        text = """
        [[rule]]
        tool = "shell_exec"
        arg_glob = { command = "rm -rf*" }
        action = "deny"

        [[rule]]
        tool = "shell_exec"
        action = "allow"
        """
        policy = parse_policy_text(text)
        assert policy.decide("shell_exec", {"command": "rm -rf /"}) == "deny"
        assert policy.decide("shell_exec", {"command": "ls"}) == "allow"

    def test_unknown_action_falls_back_to_ask(self):
        text = """
        [[rule]]
        tool = "shell_exec"
        action = "yolo"
        """
        policy = parse_policy_text(text)
        # Unknown actions are coerced to "ask" — never silently allow.
        assert policy.decide("shell_exec", {"command": "x"}) == "ask"

    def test_bad_toml_returns_builtin_defaults(self):
        # Malformed input → fall back to safe defaults rather than crash.
        policy = parse_policy_text("[[rule\n tool = ")
        # Defaults allow read-only tools and ask on writes.
        assert policy.decide("file_read", {}) == "allow"
        assert policy.decide("shell_exec", {}) == "ask"

    def test_defaults_fallback_section(self):
        text = """
        [defaults]
        fallback = "deny"
        """
        policy = parse_policy_text(text)
        assert policy.decide("unknown_tool", {}) == "deny"

    def test_explicit_http_request_rule_overrides_always_ask_floor(self):
        # Operator-authored rule for http_request takes precedence
        # over the builtin always-ask floor. Useful for whitelisting
        # localhost / internal services during testing.
        text = """
        [[rule]]
        tool = "http_request"
        arg_glob = { url = "http://localhost*" }
        action = "allow"
        """
        policy = parse_policy_text(text)
        assert policy.decide(
            "http_request", {"url": "http://localhost:8080/foo"},
        ) == "allow"
        # Non-localhost http_request still gets asked (always-ask floor).
        assert policy.decide(
            "http_request", {"url": "https://api.example.com/data"},
        ) == "ask"


# ---------------------------------------------------------------------------
# Batch-arg (paths=[...]) vs scalar path globs
# ---------------------------------------------------------------------------

class TestBatchArgGlobs:
    """A path-scoped rule must govern the batch spelling of the same arg —
    otherwise file_read paths=[...] silently bypasses path deny rules."""

    def _policy(self, action):
        return Policy(
            rules=[_Rule(
                tool="file_read",
                arg_glob={"path": "/workspace/secrets/*"},
                action=action,
            )],
            tool_defaults={"file_read": "allow"},
            fallback="ask",
        )

    def test_deny_rule_claims_batch_when_any_path_matches(self):
        policy = self._policy("deny")
        assert policy.decide(
            "file_read",
            {"paths": ["/workspace/app.py", "/workspace/secrets/key.pem"]},
        ) == "deny"

    def test_deny_rule_ignores_batch_with_no_matching_path(self):
        policy = self._policy("deny")
        assert policy.decide(
            "file_read",
            {"paths": ["/workspace/app.py", "/workspace/README.md"]},
        ) == "allow"  # tool default

    def test_allow_rule_requires_all_paths_to_match(self):
        policy = Policy(
            rules=[_Rule(
                tool="file_read",
                arg_glob={"path": "/workspace/docs/*"},
                action="allow",
            )],
            tool_defaults={"file_read": "ask"},
            fallback="ask",
        )
        # Fully covered batch → the allow rule claims it.
        assert policy.decide(
            "file_read",
            {"paths": ["/workspace/docs/a.md", "/workspace/docs/b.md"]},
        ) == "allow"
        # Partially covered batch → falls through to the ask default.
        assert policy.decide(
            "file_read",
            {"paths": ["/workspace/docs/a.md", "/workspace/app.py"]},
        ) == "ask"

    def test_scalar_path_rule_behaviour_unchanged(self):
        policy = self._policy("deny")
        assert policy.decide(
            "file_read", {"path": "/workspace/secrets/key.pem"},
        ) == "deny"
        assert policy.decide(
            "file_read", {"path": "/workspace/app.py"},
        ) == "allow"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestPolicyRoundTrip:
    def test_to_toml_then_parse_preserves_decisions(self):
        original = Policy(
            rules=[
                _Rule(tool="file_read", action="allow"),
                _Rule(
                    tool="shell_exec",
                    arg_glob={"command": "git *"},
                    action="allow",
                ),
                _Rule(
                    tool="shell_exec",
                    arg_glob={"command": "rm -rf*"},
                    action="deny",
                ),
            ],
            fallback="ask",
        )
        text = policy_to_toml(original)
        round_trip = parse_policy_text(text)
        # Same decisions on the canonical test cases.
        assert round_trip.decide("file_read", {}) == "allow"
        assert round_trip.decide("shell_exec", {"command": "git status"}) == "allow"
        assert round_trip.decide("shell_exec", {"command": "rm -rf /"}) == "deny"
        assert round_trip.decide("shell_exec", {"command": "ls"}) == "ask"

    def test_policy_as_dict_shape(self):
        # Serialization shape for /policy GET endpoints.
        policy = Policy(
            rules=[_Rule(tool="file_read", action="allow")],
            fallback="ask",
        )
        d = policy_as_dict(policy)
        assert d["rules"] == [{
            "tool": "file_read",
            "action": "allow",
            "arg_glob": {},
        }]
        assert d["fallback"] == "ask"
