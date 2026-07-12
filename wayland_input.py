"""
Input backends for driving osu! under wine/XWayland on a Wayland session.

Two backends, selected with DEEPOSU_INPUT (default: "xtest"):

- "xtest": absolute pointer warps + key events injected directly into the
  XWayland server (the X display wine/osu! lives on) via the XTest extension.
  This bypasses the compositor entirely, which matters: Hyprland moves its
  own cursor for virtual uinput devices but does NOT forward their motion to
  XWayland clients (verified: XWayland pointer valuators stay frozen while
  the compositor cursor tracks perfectly). XTest state is also readable back
  (query_pointer), so positioning is verifiable and exact by construction.

- "uinput": kernel-level virtual devices (absolute tablet-style pointer +
  keyboard). Compositor-true — the desktop cursor really moves — and fully
  display-server agnostic, but on Hyprland the game never receives the
  motion (see above). Kept for native-Wayland targets and future compositor
  fixes.

The kill switch always reads physical keyboards via evdev, regardless of
backend — pynput-style global listeners don't work on Wayland.
"""

import os
import time
import threading
from select import select

from evdev import InputDevice, UInput, AbsInfo, ecodes as e, list_devices

ABS_MAX = 65535

KEY_MAP = {
    's': e.KEY_S,
    'd': e.KEY_D,
    'z': e.KEY_Z,
    'x': e.KEY_X,
    'esc': e.KEY_ESC,
    'enter': e.KEY_ENTER,
    'space': e.KEY_SPACE,
    'grave': e.KEY_GRAVE,     # osu! quick-retry
    'backspace': e.KEY_BACKSPACE,
    'down': e.KEY_DOWN,       # song-select carousel navigation
    'up': e.KEY_UP,
}

# Full a-z/0-9/space map for typing search queries via uinput
KEY_MAP_TYPING = {c: getattr(e, f"KEY_{c.upper()}") for c in "abcdefghijklmnopqrstuvwxyz"}
KEY_MAP_TYPING.update({d: getattr(e, f"KEY_{d}") for d in "0123456789"})
KEY_MAP_TYPING[' '] = e.KEY_SPACE

# All key codes the virtual keyboard must advertise (game keys + typing + shift)
_ALL_KB_CODES = sorted(set(KEY_MAP.values()) | set(KEY_MAP_TYPING.values()) | {e.KEY_LEFTSHIFT})

VIRTUAL_POINTER_NAME = "osu-rl-virtual-pointer"
VIRTUAL_KEYBOARD_NAME = "osu-rl-virtual-keyboard"


# =====================================================================
# XTest backend (default) — injects into the XWayland server directly
# =====================================================================

def _open_display():
    from Xlib import display as xdisplay
    return xdisplay.Display(os.environ.get("DISPLAY", ":1"))


class XTestPointer:
    """Absolute pointer via XTest warps on the X display osu! runs on."""

    def __init__(self, screen_w: int, screen_h: int):
        from Xlib import X
        from Xlib.ext import xtest
        self._X = X
        self._xtest = xtest
        self._d = _open_display()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = screen_w // 2
        self.y = screen_h // 2

    def move_to(self, x: int, y: int):
        """Warp to absolute screen pixel coordinates (clamped to screen)."""
        self.x = max(0, min(self.screen_w - 1, int(x)))
        self.y = max(0, min(self.screen_h - 1, int(y)))
        # detail=0 -> absolute motion
        self._xtest.fake_input(self._d, self._X.MotionNotify, detail=0,
                               x=self.x, y=self.y)
        self._d.sync()

    @property
    def position(self):
        return self.x, self.y

    def verify(self):
        """Read the true X pointer position back (XTest-only capability)."""
        p = self._d.screen().root.query_pointer()
        return p.root_x, p.root_y

    def close(self):
        self._d.close()


class XTestKeyboard:
    """Key events via XTest on the X display osu! runs on."""

    def __init__(self):
        from Xlib import X, XK
        from Xlib.ext import xtest
        self._X = X
        self._xtest = xtest
        self._d = _open_display()
        keysyms = {
            's': XK.XK_s, 'd': XK.XK_d, 'z': XK.XK_z, 'x': XK.XK_x,
            'esc': XK.XK_Escape, 'enter': XK.XK_Return,
            'space': XK.XK_space, 'grave': XK.XK_grave,
            'backspace': XK.XK_BackSpace,
            'down': XK.XK_Down, 'up': XK.XK_Up,
        }
        self._keycodes = {name: self._d.keysym_to_keycode(sym)
                          for name, sym in keysyms.items()}
        self._shift = self._d.keysym_to_keycode(XK.XK_Shift_L)

    def press(self, key: str):
        self._xtest.fake_input(self._d, self._X.KeyPress, self._keycodes[key])
        self._d.sync()

    def release(self, key: str):
        self._xtest.fake_input(self._d, self._X.KeyRelease, self._keycodes[key])
        self._d.sync()

    def tap(self, key: str, hold: float = 0.05):
        self.press(key)
        time.sleep(hold)
        self.release(key)

    def type_text(self, text: str, per_char: float = 0.04):
        """Type an ASCII string (letters/digits/space) for the osu! search box.

        Uppercase is produced by holding Shift over the lowercase key; the
        keysym of a-z/0-9/space equals its ASCII code, so keysym_to_keycode
        maps directly. Unmappable characters are skipped.
        """
        for ch in text:
            shifted = ch.isupper()
            kc = self._d.keysym_to_keycode(ord(ch.lower()))
            if kc == 0:
                continue
            if shifted:
                self._xtest.fake_input(self._d, self._X.KeyPress, self._shift)
            self._xtest.fake_input(self._d, self._X.KeyPress, kc)
            self._xtest.fake_input(self._d, self._X.KeyRelease, kc)
            if shifted:
                self._xtest.fake_input(self._d, self._X.KeyRelease, self._shift)
            self._d.sync()
            time.sleep(per_char)

    def close(self):
        self._d.close()


# =====================================================================
# uinput backend — kernel virtual devices (compositor-true)
# =====================================================================

class UinputPointer:
    """Absolute-positioning virtual pointer. Tracks its own position exactly."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = screen_w // 2
        self.y = screen_h // 2
        self._ui = UInput(
            {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
                e.EV_ABS: [
                    (e.ABS_X, AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)),
                    (e.ABS_Y, AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)),
                ],
            },
            name=VIRTUAL_POINTER_NAME,
        )
        time.sleep(0.5)  # let the compositor register the new device

    def move_to(self, x: int, y: int):
        """Move to absolute screen pixel coordinates (clamped to screen)."""
        self.x = max(0, min(self.screen_w - 1, int(x)))
        self.y = max(0, min(self.screen_h - 1, int(y)))
        self._ui.write(e.EV_ABS, e.ABS_X, self.x * ABS_MAX // (self.screen_w - 1))
        self._ui.write(e.EV_ABS, e.ABS_Y, self.y * ABS_MAX // (self.screen_h - 1))
        self._ui.syn()

    @property
    def position(self):
        return self.x, self.y

    def close(self):
        self._ui.close()


class UinputKeyboard:
    """Virtual keyboard for game keys and menu navigation."""

    def __init__(self):
        self._ui = UInput(
            {e.EV_KEY: _ALL_KB_CODES},
            name=VIRTUAL_KEYBOARD_NAME,
        )
        time.sleep(0.5)  # let the compositor register the new device

    def press(self, key: str):
        self._ui.write(e.EV_KEY, KEY_MAP[key], 1)
        self._ui.syn()

    def release(self, key: str):
        self._ui.write(e.EV_KEY, KEY_MAP[key], 0)
        self._ui.syn()

    def tap(self, key: str, hold: float = 0.05):
        self.press(key)
        time.sleep(hold)
        self.release(key)

    def type_text(self, text: str, per_char: float = 0.04):
        """Type an ASCII string via uinput (letters/digits/space)."""
        for ch in text:
            shifted = ch.isupper()
            code = KEY_MAP_TYPING.get(ch.lower())
            if code is None:
                continue
            if shifted:
                self._ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1)
            self._ui.write(e.EV_KEY, code, 1)
            self._ui.write(e.EV_KEY, code, 0)
            if shifted:
                self._ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 0)
            self._ui.syn()
            time.sleep(per_char)

    def close(self):
        self._ui.close()


# =====================================================================
# Backend selection — osu_env imports these names
# =====================================================================

def _backend() -> str:
    return os.environ.get("DEEPOSU_INPUT", "xtest").lower()


def VirtualPointer(screen_w: int, screen_h: int):
    if _backend() == "uinput":
        return UinputPointer(screen_w, screen_h)
    return XTestPointer(screen_w, screen_h)


def VirtualKeyboard():
    if _backend() == "uinput":
        return UinputKeyboard()
    return XTestKeyboard()


# =====================================================================
# Kill switch — always evdev on physical keyboards
# =====================================================================

class KillSwitchListener:
    """
    Watches real (physical) keyboards for a hotkey via evdev.

    pynput's global Listener cannot capture keys under Wayland; reading
    /dev/input/event* directly works regardless of compositor (requires the
    user to be in the `input` group).
    """

    def __init__(self, callback, key_code=e.KEY_RIGHTBRACE):
        self.callback = callback
        self.key_code = key_code
        self.devices = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            if dev.name in (VIRTUAL_POINTER_NAME, VIRTUAL_KEYBOARD_NAME):
                dev.close()
                continue
            if key_code in dev.capabilities().get(e.EV_KEY, []):
                self.devices.append(dev)
            else:
                dev.close()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self.devices:
            print("[KillSwitch] WARNING: no readable keyboard found — "
                  "kill switch disabled. Is your user in the `input` group?")
            return self
        names = ", ".join(d.name for d in self.devices)
        print(f"[KillSwitch] Listening on: {names}")
        self._thread.start()
        return self

    def _run(self):
        dev_map = {dev.fd: dev for dev in self.devices}
        while not self._stop:
            readable, _, _ = select(dev_map, [], [], 0.5)
            for fd in readable:
                try:
                    for event in dev_map[fd].read():
                        if (event.type == e.EV_KEY and
                                event.code == self.key_code and
                                event.value == 1):
                            self.callback()
                except OSError:
                    pass

    def stop(self):
        self._stop = True


if __name__ == "__main__":
    print(f"Backend: {_backend()}")
    ptr = VirtualPointer(2560, 1440)
    kb = VirtualKeyboard()
    print(f"Pointer starting at {ptr.position}, tracing a square in 2s...")
    time.sleep(2)
    ok = True
    for tx, ty in [(800, 400), (1800, 400), (1800, 1000), (800, 1000)]:
        ptr.move_to(tx, ty)
        time.sleep(0.3)
        if hasattr(ptr, "verify"):
            rx, ry = ptr.verify()
            good = (rx, ry) == (tx, ty)
            ok &= good
            print(f"  move_to({tx},{ty}) -> X pointer ({rx},{ry}) {'OK' if good else 'MISMATCH'}")
    print("PASS — input backend works" if ok else
          "Square traced (no readback on this backend) — did the cursor move?")
    ptr.close()
    kb.close()
