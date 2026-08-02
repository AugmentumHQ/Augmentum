"""Universal store converters — real specimen files from each dialect
(fetched 2026-07-19) through conversion + our actual manifest gate."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from augmentum.marketplace.converters import (
    convert_casaos,
    convert_runtipi,
    convert_umbrel,
)
from augmentum.marketplace.converters.base import classify_env, image_is_pinned
from augmentum.marketplace.manifest import parse_manifest

DATA = Path(__file__).parent / "data" / "store_specimens"


def _read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


class TestRuntipi:
    def test_actual_budget_converts_and_passes_gate(self):
        res = convert_runtipi(
            json.loads(_read("runtipi-actual-config.json")),
            _read("runtipi-actual-compose.yml"),
        )
        assert res.eligible, res.reasons
        svc = res.listing["install_payload"]["service"]
        assert svc["image"] == "actualbudget/actual-server:26.7.0"
        assert svc["port"] == 5006          # x-runtipi internal_port, NOT host 8011
        assert svc["volumes"] == {"data": "/data"}
        parse_manifest(res.listing["install_payload"])
        # Browser block is a default → must be flagged for review.
        assert any("browser block" in r for r in res.review)

    def test_host_port_fallback_is_flagged(self):
        cfg = {"id": "x", "name": "X", "port": 8123, "categories": []}
        compose = "services:\n  x:\n    image: a/b:1.0\n"
        res = convert_runtipi(cfg, compose)
        assert res.eligible
        assert res.listing["install_payload"]["service"]["port"] == 8123
        assert any("HOST side" in r for r in res.review)

    def test_container_port_from_old_style_mapping(self):
        cfg = {"id": "x", "name": "X", "port": 8129, "categories": []}
        compose = ("services:\n  x:\n    image: a/b:1.0\n"
                   "    ports:\n      - ${APP_PORT}:80\n")
        res = convert_runtipi(cfg, compose)
        assert res.listing["install_payload"]["service"]["port"] == 80


class TestCasaos:
    def test_official_actualbudget(self):
        res = convert_casaos(_read("casaos-actual-compose.yml"),
                             repo_slug="casaos-official")
        assert res.eligible, res.reasons
        svc = res.listing["install_payload"]["service"]
        assert svc["port"] == 5006
        assert res.listing["install_payload"]["resources"]["ram_mb"] == 128
        parse_manifest(res.listing["install_payload"])

    def test_bigbear_it_tools_digest_stripped_and_id_collapsed(self):
        res = convert_casaos(_read("bigbear-ittools-compose.yml"),
                             repo_slug="big-bear")
        assert res.eligible, res.reasons
        assert res.app_id == "it-tools"          # big-bear- prefix collapsed
        svc = res.listing["install_payload"]["service"]
        assert "@sha256" not in svc["image"]
        assert svc["port"] == 80                 # container side, not host 8080
        parse_manifest(res.listing["install_payload"])


class TestUmbrel:
    def test_dependency_apps_are_ineligible(self):
        res = convert_umbrel(
            {"id": "electrs", "name": "Electrs", "dependencies": ["bitcoin"]},
            "services:\n  app:\n    image: a/b:1.0\n",
        )
        assert not res.eligible
        assert any("depends on" in r for r in res.reasons)

    def test_app_proxy_is_ignored_and_port_read_from_it(self):
        compose = yaml.safe_dump({
            "services": {
                "app_proxy": {"environment": {"APP_PORT": 8384,
                                              "APP_HOST": "x"}},
                "server": {"image": "syncthing/syncthing:2.1.2",
                           "volumes": ["${APP_DATA_DIR}/data:/var/syncthing"]},
            },
        })
        res = convert_umbrel(
            {"id": "syncthing", "name": "Syncthing", "category": "files"},
            compose,
        )
        assert res.eligible, res.reasons
        svc = res.listing["install_payload"]["service"]
        assert svc["port"] == 8384
        assert svc["volumes"] == {"syncthing": "/var/syncthing"}
        parse_manifest(res.listing["install_payload"])


class TestGateRules:
    def test_unpinned_tags_refused(self):
        assert not image_is_pinned("a/b:latest")
        assert not image_is_pinned("a/b")
        assert image_is_pinned("a/b:1.2.3")
        assert image_is_pinned("a/b@sha256:abc")

    def test_docker_socket_refused(self):
        res = convert_casaos(
            "name: bad\nservices:\n  bad:\n    image: a/b:1.0\n"
            "    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n"
            "x-casaos:\n  main: bad\n  port_map: '80'\n",
        )
        assert not res.eligible
        assert any("docker socket" in r for r in res.reasons)

    def test_template_env_never_leaks(self):
        safe, prompts, review = classify_env({
            "PLAIN": "1",
            "TZ": "${TZ}",
            "PW": "$APP_PASSWORD",
            "HOST": "${DEVICE_DOMAIN_NAME}",
        })
        assert safe == {"PLAIN": "1"}
        assert [p["key"] for p in prompts] == ["PW"]
        assert len(review) == 2
