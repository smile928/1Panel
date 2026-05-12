# AGENTS.md

## Cursor Cloud specific instructions

### Architecture Overview
1Panel is a VPS control panel with a **Core + Agent** architecture:
- **`agent/`** — Go data-plane (Docker, websites, DBs, firewall, AI). Listens on Unix socket `/etc/1panel/agent.sock`.
- **`core/`** — Go control-plane (auth, sessions, settings). Serves HTTP on port 9999 and proxies to agent.
- **`frontend/`** — Vue 3 SPA (Vite on port 4004 in dev, proxies `/api/v2` → `localhost:9999`).

### Prerequisites (already in update script / environment)
- Go 1.25.7 at `/usr/local/go` (symlinked to `/usr/bin/go`)
- Node.js 22.x + npm
- Docker CE (with fuse-overlayfs + iptables-legacy for nested container env)

### Dev Config Files (must exist before running backends)
- `/opt/1panel/conf/app.yaml` — dev-mode config (triggers `mode: dev` path in viper init)
- `/usr/local/bin/1pctl` — stub with `BASE_DIR=/opt`, `ORIGINAL_VERSION=v2.0.0`, `ORIGINAL_PORT=9999`, `ORIGINAL_USERNAME=admin`, `ORIGINAL_PASSWORD=admin123`, `LANGUAGE=en`
- Directories: `/opt/1panel/{conf,db,log,tmp}`, `/etc/1panel/`, `/.1panel_clash`

### Running Services

**Start order matters:** agent first, then core, then frontend.

```bash
# 1. Agent (must run as root for socket + Docker access)
cd /workspace/agent && sudo /tmp/1panel-agent

# 2. Core (must run as root to connect to agent socket)
cd /workspace/core && sudo /tmp/1panel-core

# 3. Frontend dev server
cd /workspace/frontend && npm run dev
```

### Building

```bash
# Build agent
cd /workspace/agent && CGO_ENABLED=0 go build -trimpath -o /tmp/1panel-agent ./cmd/server/main.go

# Build core (requires web assets — create stubs if not doing a full frontend build)
cd /workspace/core && CGO_ENABLED=0 go build -trimpath -o /tmp/1panel-core ./cmd/server/main.go
```

**Core compile note:** The core embeds `core/cmd/server/web/{index.html,assets/*,favicon.png,static/*}`. For dev, create minimal stub files in that directory. For production build, run `cd frontend && npm run build:pro` which outputs to `core/cmd/server/web/`.

### Lint & Tests

```bash
# Frontend lint
cd /workspace/frontend && npx eslint --ext .js,.ts,.vue ./src

# Go vet
cd /workspace/core && go vet ./...
cd /workspace/agent && go vet ./...

# Type check (slow, ~2min)
cd /workspace/frontend && npm run type-check
```

### Login Credentials (dev mode)
- Username: `admin`
- Password: `admin123`
- URL: http://localhost:4004/

### Gotchas
- The agent **panics** if `/usr/local/bin/1pctl` is missing or lacks `BASE_DIR`/`ORIGINAL_VERSION` entries.
- Docker daemon must be running before starting the agent (many features depend on Docker SDK).
- Start dockerd with `sudo dockerd` if it's not running as a service.
- The agent creates `/.1panel_clash` at filesystem root (recycle bin) — needs write permission.
- Both agent and core must run as **root** (socket permissions, Docker access, firewall management).
