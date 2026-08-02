---
name: Bug report
about: Something is broken or behaving wrong
title: "[bug] "
labels: bug
assignees: ''
---

## What happened

A clear description of the bug.

## Steps to reproduce

1.
2.
3.

## Expected behavior

What you expected to happen instead.

## Actual behavior

What actually happened. Include error messages, stack traces, and relevant log lines
(`docker compose logs augmentum`). Set `AUGMENTUM_LOG_LEVEL=DEBUG` for more detail.

## Environment

- Augmentum version / commit:
- Install method: [ ] installer script  [ ] GHCR pull (`docker compose pull`)  [ ] cloned repo + local build
- Variant: [ ] cpu  [ ] gpu
- Compose overlays enabled (from `.augmentum.conf`): e.g. `compose.gpu.yaml`, `compose.kokoro.yaml`
- Backend in use: [ ] bundled engine  [ ] Ollama  [ ] OpenAI-compatible  [ ] Anthropic  [ ] other:
- Model:
- Host OS:
- Docker / Docker Compose version (`docker version`, `docker compose version`):
- GPU (if relevant): make/model + driver version
- Browser (for UI bugs): name + version
- Deployment: [ ] localhost only  [ ] LAN (`AUGMENTUM_BIND_HOST=0.0.0.0`)  [ ] behind reverse proxy

## Screenshots / recordings

If applicable.

## Anything else

Other context, things you've already tried, etc.
