#!/usr/bin/env bash
# Re-map a minimized osu! window. Wine (>=11.13) minimizes exclusive-fullscreen
# games on focus loss, unmapping the window entirely — this brings it back.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$REPO_DIR/.venv/bin/python" - <<'EOF'
from Xlib import display as xd
import time
d = xd.Display(":1")
for c in d.screen().root.query_tree().children:
    try:
        cls = c.get_wm_class()
        name = c.get_wm_name() or ""
        if cls and 'osu' in str(cls).lower() and name.startswith("osu!"):
            c.map(); d.sync(); time.sleep(0.5)
            state = c.get_attributes().map_state
            print(f"mapped {name!r}: {'restored' if state == 2 else f'map_state={state}'}")
            break
    except Exception:
        pass
else:
    print("no osu! window found — is the game running?")
d.close()
EOF
