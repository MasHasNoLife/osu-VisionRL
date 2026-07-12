#!/usr/bin/env bash
# Bootstraps a full Deeposu training session: virtual camera, screen feed,
# tosu telemetry, and osu! under wine. Idempotent — safe to re-run; each
# component is skipped if already up.
set -euo pipefail

PREFIX="$HOME/.osu-wine"
OSU_EXE="$PREFIX/drive_c/users/$USER/AppData/Local/osu!/osu!.exe"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Personal overrides (e.g. OSU_ARGS="-devserver ...") live in an untracked
# scripts/session.env so they never end up in the public repo.
[ -f "$REPO_DIR/scripts/session.env" ] && . "$REPO_DIR/scripts/session.env"
OSU_ARGS="${OSU_ARGS:-}"

mkdir -p "$REPO_DIR/logs"

# 0. Post-update guard: if the running kernel's module dir is gone, the system
#    was updated (kernel replaced) and NOTHING kernel-side works right until a
#    reboot — v4l2loopback can't load and the GPU driver will be mismatched.
if [ ! -d "/lib/modules/$(uname -r)" ]; then
    echo "ERROR: running kernel $(uname -r) has no module directory."
    echo "A system update replaced the kernel. REBOOT first, then re-run this."
    exit 1
fi

# 1. v4l2loopback on /dev/video9
if [ ! -e /dev/video9 ]; then
    echo "[1/4] Loading v4l2loopback (needs sudo)..."
    sudo modprobe v4l2loopback video_nr=9 card_label="VirtualCam" exclusive_caps=1
else
    echo "[1/4] /dev/video9 present"
fi

# 2. wf-recorder feeding the virtual camera
if ! pgrep -f "wf-recorder.*video9" >/dev/null; then
    echo "[2/4] Starting wf-recorder..."
    wf-recorder -c rawvideo -m v4l2 -x yuv420p -f /dev/video9 \
        >"$REPO_DIR/logs/wf-recorder.log" 2>&1 &
    disown
else
    echo "[2/4] wf-recorder already running"
fi

# 3. tosu — Windows build, must run in the SAME wine prefix as osu! so
#    ReadProcessMemory can reach the osu! process via the shared wineserver.
#    cwd must be the repo dir so tosu picks up tosu.env (POLL_RATE etc).
if ! pgrep -f "tosu.exe" >/dev/null; then
    echo "[3/4] Starting tosu under wine..."
    (cd "$REPO_DIR" && WINEPREFIX="$PREFIX" wine ./tosu.exe \
        >"$REPO_DIR/logs/tosu-runtime.log" 2>&1 &)
else
    echo "[3/4] tosu already running"
fi

# 4. osu!
if ! pgrep -f "osu!.exe" >/dev/null; then
    echo "[4/4] Starting osu!..."
    WINEPREFIX="$PREFIX" wine "$OSU_EXE" $OSU_ARGS \
        >"$REPO_DIR/logs/osu.log" 2>&1 &
    disown
else
    echo "[4/4] osu! already running"
fi

echo
echo -n "Waiting for tosu websocket (127.0.0.1:24050) "
for _ in $(seq 1 30); do
    if timeout 1 bash -c 'echo > /dev/tcp/127.0.0.1/24050' 2>/dev/null; then
        echo "— UP."
        echo
        echo "Session ready. Select a map in osu!, then run:"
        echo "    python train_deeposu.py"
        exit 0
    fi
    echo -n "."
    sleep 1
done
echo
echo "WARNING: tosu websocket did not come up within 30s — check logs/tosu-runtime.log"
exit 1
