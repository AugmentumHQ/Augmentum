# Operate exemplar

The shape of an operate turn: start a service, verify it's actually
reachable, report the URL that works. A URL printed by the service is
NOT the same as a URL the user can reach — confirm with a probe before
claiming success.

## User goal

"Run the FastAPI app and give me a URL I can hit from my browser."

## Turn

[file_read pyproject.toml]
  → entry point: augmentum.proxy.server:create_app
  → uvicorn is a dependency

[service_start "uvicorn augmentum.proxy.server:create_app --factory --host 0.0.0.0 --port 8000"]
  → service id: svc-7a3b
  → log line: "Uvicorn running on http://0.0.0.0:8000"

# "Running on" is what the SERVICE thinks. The URL the user can reach
# depends on what ports the container publishes to the host.

[container_info]
  → published ports: 8000 → 8000 (host)
  → container_ip: 172.18.0.5

[service_probe svc-7a3b "/"]
  → HTTP 200 OK in 23ms
  → Content-Type: text/html
  → body looks like the expected landing page

# Reachable from inside the container. Confirm host-side too.

[shell_exec "curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/"]
  → 200

# Verified end-to-end. Report the URL.

Answer to user:

The app is running. Reach it at: **http://localhost:8000/**

Verified with a probe (HTTP 200, 23ms). Container port 8000 is
published to host port 8000. Service id `svc-7a3b` — stop with
`service_stop` when you're done.

[finish_task]

## What this exemplar teaches

- A URL printed by a service is not a URL the user can reach.
  Containers have their own network namespace; the host sees a
  different address. Always probe before reporting.
- Use `service_start` for long-running processes, not `shell_exec`
  with `&`. The runtime tracks lifecycle and gives you a stable id.
- `container_info` shows the host-visible port mapping. Use it before
  reporting any URL.
- `service_probe` verifies the service responds from inside the
  container. `curl` from `shell_exec` verifies it's reachable from
  the host. Both checks matter when the user's goal is "I can hit
  this from my browser."
- Don't claim "the URL is X" until you've confirmed X actually returns
  a response. "The service is running at http://0.0.0.0:8000, try it"
  without a probe is failure dressed up as success.
