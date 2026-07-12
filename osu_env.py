"""
Osu! Gymnasium Environment v3.
- Absolute-aim action space: MultiDiscrete([64, 48, 2]) = X bin x Y bin x click.
  No cursor speed cap — the same action space scales from 1-star to 8-star maps.
- Dict observation: 4 stacked 96x96 frames + 8-dim game-state vector
  (cursor position, next-object geometry from tosu + beatmap parsing)
- Potential-based closeness shaping via beatmap parsing (corner-bias free)
- Combo-break (slider break) penalty from live telemetry
- Input via kernel uinput devices (Wayland native, exact cursor tracking)
"""

import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import json
import math
import threading
import websocket
from collections import deque

# cv2 bundles a Qt build with no Wayland plugin and no fonts, and its import
# points QT_QPA_FONTDIR at a nonexistent bundled dir. Qt reads these only at
# first window creation, so forcing them here (after cv2 import, before the
# first imshow) silences the warnings. The debug window runs via XWayland.
os.environ["QT_QPA_PLATFORM"] = "xcb"
for _font_dir in ("/usr/share/fonts/liberation", "/usr/share/fonts/noto"):
    if os.path.isdir(_font_dir):
        os.environ["QT_QPA_FONTDIR"] = _font_dir
        break

from beatmap_parser import BeatmapParser
from wayland_input import VirtualPointer, VirtualKeyboard


class OsuEnv(gym.Env):
    """osu! environment with absolute aim and dense rewards."""

    metadata = {"render_modes": ["human"]}

    # Absolute aim grid (must match agent.Agent bins)
    X_BINS = 64
    Y_BINS = 48

    STATE_DIM = 8
    FRAME_STACK = 4
    SONGS_DIR = "/home/mas/.osu-wine/drive_c/users/mas/AppData/Local/osu!/Songs"

    # Step pacing: cap the control loop at 60 Hz so gamma horizon and click
    # granularity stay consistent even if the capture source runs faster.
    STEP_DT = 1.0 / 60.0

    # Aim shaping: small per-step bonus for cursor proximity to the target.
    # Deliberately NOT potential-based: potential shaping is advantage-
    # invariant (Wiewiora 2003) — the critic absorbs the potential (trivially
    # here, since the state vector contains the target offset) and the aim
    # signal vanishes from the advantages. A direct proximity bonus keeps a
    # persistent advantage gap between aiming at the target and anywhere
    # else. Safe with absolute aim: the bonus is maximized exactly by
    # correct play (hover the target), and steps with no upcoming object are
    # unshaped, so there is no stray attractor (the v1 corner-bias mode).
    # Scale: ~53 steps/object at 60Hz -> perfect hover earns ~2.6 per
    # object. Raised from 0.02 after live logs showed the dense aim signal
    # drowned by hit-spike variance in normalized advantages (accuracy rose
    # via hits while average proximity stayed at random level).
    PROXIMITY_COEF = 0.05

    # Combo drop without a new miss = slider break
    SLIDER_BREAK_PENALTY = -0.3

    # Watchdog: if song time stops advancing this long while "playing",
    # the game is frozen (wine deadlock) — truncate the episode.
    STALL_TIMEOUT_S = 15.0

    def __init__(self):
        super().__init__()

        # ===== ACTION SPACE =====
        self.action_space = spaces.MultiDiscrete([self.X_BINS, self.Y_BINS, 2])

        # ===== OBSERVATION SPACE =====
        self.observation_space = spaces.Dict({
            'frames': spaces.Box(low=0, high=255,
                                 shape=(self.FRAME_STACK, 96, 96), dtype=np.uint8),
            'state': spaces.Box(low=-2.0, high=2.0,
                                shape=(self.STATE_DIM,), dtype=np.float32),
        })

        # ===== VIDEO CAPTURE =====
        self.cap = cv2.VideoCapture('/dev/video9')
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = self.cap.read()
        if ret:
            self.SCREEN_H, self.SCREEN_W = frame.shape[:2]
        else:
            self.SCREEN_H, self.SCREEN_W = 1440, 2560
            print("[WARNING] Could not read initial frame, using default 2560x1440")

        # ===== PLAYFIELD BOUNDS =====
        self.PLAY_TOP = int(self.SCREEN_H * 0.05)
        self.PLAY_BOTTOM = int(self.SCREEN_H * 0.95)
        self.PLAY_LEFT = int(self.SCREEN_W * 0.145)
        self.PLAY_RIGHT = int(self.SCREEN_W * 0.855)

        self.PLAY_W = self.PLAY_RIGHT - self.PLAY_LEFT
        self.PLAY_H = self.PLAY_BOTTOM - self.PLAY_TOP

        # Playfield diagonal (for normalizing distances)
        self.PLAYFIELD_DIAG = math.hypot(self.PLAY_W, self.PLAY_H)

        # ===== INPUT DEVICES (kernel uinput — Wayland native) =====
        # Absolute pointer: position is set, never read back, so it can't drift.
        self.pointer = VirtualPointer(self.SCREEN_W, self.SCREEN_H)
        self.keyboard = VirtualKeyboard()

        # ===== FRAME STACK =====
        self.frame_stack = deque(maxlen=self.FRAME_STACK)

        # ===== TOSU STATE =====
        self.is_playing = False
        self.current_score = 0
        self.current_combo = 0
        self.current_misses = 0
        self.current_300s = 0
        self.current_100s = 0
        self.current_50s = 0
        self.current_hp = 1.0
        self.current_time_ms = 0
        self.map_length_ms = 1
        self.max_combo = 1

        # Previous frame state (for delta reward)
        self.prev_misses = 0
        self.prev_300s = 0
        self.prev_100s = 0
        self.prev_50s = 0
        self.prev_combo = 0

        # High-water marks
        self.max_score = 0
        self.max_combo_achieved = 0
        self.max_misses = 0
        self.max_300s_achieved = 0
        self.max_100s_achieved = 0
        self.max_50s_achieved = 0

        # Map metadata
        self.map_folder = ""
        self.map_file = ""

        # ===== BEATMAP PARSER =====
        self.beatmap = BeatmapParser(self.SONGS_DIR)
        self.beatmap_loaded = False

        # ===== CLICK STATE =====
        self.current_key = 's'
        self.is_held = False

        # ===== TRACKING =====
        self.current_episode_reward = 0.0
        self.total_steps = 0
        self._step_t = time.monotonic()
        self._rate_t = time.monotonic()
        self._last_time_ms = -1
        self._last_time_wall = time.monotonic()

        # ===== REWARD SHAPING STATE =====
        self.episode_shaping = 0.0

        # ===== MAP ROTATION =====
        # Set by the trainer before reset() to switch maps; consumed once.
        self.pending_map_query = None

        # ===== LOGGING =====
        self.show_debug = os.environ.get("DEEPOSU_DEBUG", "1") != "0"
        self.image_logs_dir = "image_logs"
        os.makedirs(self.image_logs_dir, exist_ok=True)
        self.last_image_log_time = 0

        # ===== START WEBSOCKET =====
        self._start_websocket()
        print(f"[OsuEnv] Initialized. Screen: {self.SCREEN_W}x{self.SCREEN_H}")
        print(f"[OsuEnv] Playfield: L={self.PLAY_LEFT} R={self.PLAY_RIGHT} T={self.PLAY_TOP} B={self.PLAY_BOTTOM}")
        print(f"[OsuEnv] Actions: absolute aim {self.X_BINS}x{self.Y_BINS} bins "
              f"({self.PLAY_W // self.X_BINS}x{self.PLAY_H // self.Y_BINS}px/bin) x 2 click states")
        print(f"[OsuEnv] Input: uinput absolute pointer + virtual keyboard (Wayland native)")
        print(f"[OsuEnv] Shaping: direct proximity bonus, coef={self.PROXIMITY_COEF}/step")

    def _start_websocket(self):
        """Connect to Tosu websocket in a background thread."""
        def on_message(ws, message):
            try:
                data = json.loads(message)

                menu_state = data.get("menu", {}).get("state", 0)
                self.is_playing = (menu_state == 2)

                gp = data.get("gameplay", {})
                self.current_score = gp.get("score", 0)
                self.current_combo = gp.get("combo", {}).get("current", 0)

                hits = gp.get("hits", {})
                self.current_misses = hits.get("0", 0)
                self.current_300s = hits.get("300", 0)
                self.current_100s = hits.get("100", 0)
                self.current_50s = hits.get("50", 0)

                hp = gp.get("hp", {})
                self.current_hp = hp.get("normal", 200.0) / 200.0

                bm = data.get("menu", {}).get("bm", {})
                time_data = bm.get("time", {})
                self.current_time_ms = time_data.get("current", 0)
                self.map_length_ms = max(time_data.get("full", 1), 1)

                path = bm.get("path", {})
                new_folder = path.get("folder", "")
                new_file = path.get("file", "")

                stats = bm.get("stats", {})
                self.max_combo = max(stats.get("maxCombo", 1), 1)

                if new_file and (new_folder != self.map_folder or new_file != self.map_file):
                    self.map_folder = new_folder
                    self.map_file = new_file
                    self.beatmap_loaded = self.beatmap.load_beatmap(new_folder, new_file)

            except Exception:
                pass

        def on_error(ws, error):
            pass

        def run_ws():
            # Reconnect loop: run_forever returns on disconnect (e.g. tosu
            # restarted), so just retry — no recursive thread spawning.
            while True:
                ws = websocket.WebSocketApp(
                    "ws://localhost:24050/ws",
                    on_message=on_message,
                    on_error=on_error,
                )
                ws.run_forever()
                time.sleep(2)

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

    def _get_single_frame(self) -> np.ndarray:
        """Capture and preprocess a single 96x96 grayscale frame."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((96, 96), dtype=np.uint8)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Mask HUD areas
            gray[0:self.PLAY_TOP, :] = 0
            gray[self.PLAY_BOTTOM:self.SCREEN_H, :] = 0
            gray[:, 0:self.PLAY_LEFT] = 0
            gray[:, self.PLAY_RIGHT:self.SCREEN_W] = 0

            # Crop and resize
            playfield = gray[self.PLAY_TOP:self.PLAY_BOTTOM, self.PLAY_LEFT:self.PLAY_RIGHT]
            resized = cv2.resize(playfield, (96, 96))

            # Debug window (disable with DEEPOSU_DEBUG=0, e.g. overnight runs)
            if self.show_debug:
                debug_view = cv2.resize(resized, (288, 288), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Deeposu v3 Vision", debug_view)
                cv2.waitKey(1)

            # Periodic logging
            current_time = time.time()
            if current_time - self.last_image_log_time >= 300:
                self.last_image_log_time = current_time
                filename = os.path.join(self.image_logs_dir, f"vision_{int(current_time)}.png")
                cv2.imwrite(filename, resized)

            return resized

        except Exception as e:
            print(f"[OsuEnv] Vision error: {e}")
            return np.zeros((96, 96), dtype=np.uint8)

    def _get_game_state(self) -> np.ndarray:
        """
        8-dim state vector, all values in [-2, 2]:
        [0] cursor x in playfield (0..1)
        [1] cursor y in playfield (0..1)
        [2] next object dx from cursor, normalized by playfield width (-1..1)
        [3] next object dy from cursor, normalized by playfield height (-1..1)
        [4] time until next object in seconds (0..2)
        [5] object-after-next dx from next object (-1..1)
        [6] object-after-next dy from next object (-1..1)
        [7] click currently held (0/1)
        """
        s = np.zeros(self.STATE_DIM, dtype=np.float32)
        s[0] = (self.pointer.x - self.PLAY_LEFT) / self.PLAY_W
        s[1] = (self.pointer.y - self.PLAY_TOP) / self.PLAY_H
        s[7] = 1.0 if self.is_held else 0.0

        if self.beatmap_loaded and self.is_playing:
            upcoming = self.beatmap.get_upcoming(self.current_time_ms, 2)
            if upcoming:
                o1 = upcoming[0]
                o1x, o1y = self.beatmap.get_screen_coords(
                    o1['x'], o1['y'],
                    self.PLAY_LEFT, self.PLAY_RIGHT, self.PLAY_TOP, self.PLAY_BOTTOM
                )
                s[2] = np.clip((o1x - self.pointer.x) / self.PLAY_W, -1.0, 1.0)
                s[3] = np.clip((o1y - self.pointer.y) / self.PLAY_H, -1.0, 1.0)
                s[4] = np.clip((o1['time'] - self.current_time_ms) / 1000.0, 0.0, 2.0)

                if len(upcoming) > 1:
                    o2 = upcoming[1]
                    o2x, o2y = self.beatmap.get_screen_coords(
                        o2['x'], o2['y'],
                        self.PLAY_LEFT, self.PLAY_RIGHT, self.PLAY_TOP, self.PLAY_BOTTOM
                    )
                    s[5] = np.clip((o2x - o1x) / self.PLAY_W, -1.0, 1.0)
                    s[6] = np.clip((o2y - o1y) / self.PLAY_H, -1.0, 1.0)

        return s

    def _get_observation(self) -> dict:
        """Get observation: stacked frames + game-state vector."""
        frame = self._get_single_frame()
        self.frame_stack.append(frame)
        return {
            'frames': np.array(self.frame_stack, dtype=np.uint8),
            'state': self._get_game_state(),
        }

    def _target_proximity(self):
        """1 − (distance from cursor to next hit object)/diagonal, or None."""
        if not (self.beatmap_loaded and self.is_playing):
            return None
        obj = self.beatmap.get_next_object(self.current_time_ms)
        if obj is None:
            return None
        ox, oy = self.beatmap.get_screen_coords(
            obj['x'], obj['y'],
            self.PLAY_LEFT, self.PLAY_RIGHT, self.PLAY_TOP, self.PLAY_BOTTOM
        )
        dist = math.hypot(self.pointer.x - ox, self.pointer.y - oy)
        return 1.0 - dist / self.PLAYFIELD_DIAG

    def _compute_reward(self) -> float:
        """
        Per-step reward:
        1. Hit: +1.0 / +0.5 / +0.2 for new 300/100/50
        2. Miss: -0.5 per new miss
        3. Slider break: -0.3 when combo drops without a new miss
        4. Proximity bonus: +PROXIMITY_COEF · (1 − normalized distance to the
           next object) while a target exists (see class comment for why this
           is direct rather than potential-based).
        """
        reward = 0.0

        # Delta hits/misses
        new_300s = self.current_300s - self.prev_300s
        new_100s = self.current_100s - self.prev_100s
        new_50s = self.current_50s - self.prev_50s
        new_misses = self.current_misses - self.prev_misses

        reward += new_300s * 1.0
        reward += new_100s * 0.5
        reward += new_50s * 0.2
        reward += new_misses * -0.5

        # Slider break: combo dropped but no miss was recorded
        if self.is_playing and new_misses == 0 and self.current_combo < self.prev_combo:
            reward += self.SLIDER_BREAK_PENALTY

        if new_300s > 0 or new_100s > 0 or new_50s > 0:
            print(f"  HIT! 300s:{self.current_300s} 100s:{self.current_100s} "
                  f"50s:{self.current_50s} misses:{self.current_misses} "
                  f"combo:{self.current_combo}")

        # Update previous state
        self.prev_300s = self.current_300s
        self.prev_100s = self.current_100s
        self.prev_50s = self.current_50s
        self.prev_misses = self.current_misses
        self.prev_combo = self.current_combo

        # Proximity bonus toward the current aim target
        proximity = self._target_proximity()
        if proximity is not None:
            shaping = self.PROXIMITY_COEF * proximity
            reward += shaping
            self.episode_shaping += shaping

        return reward

    def step(self, action):
        """Execute one step. action = [x_bin, y_bin, click_state]."""
        # High-water marks
        if self.is_playing:
            self.max_score = max(self.max_score, self.current_score)
            self.max_combo_achieved = max(self.max_combo_achieved, self.current_combo)
            self.max_misses = max(self.max_misses, self.current_misses)
            self.max_300s_achieved = max(self.max_300s_achieved, self.current_300s)
            self.max_100s_achieved = max(self.max_100s_achieved, self.current_100s)
            self.max_50s_achieved = max(self.max_50s_achieved, self.current_50s)

        # Episode end check
        if not self.is_playing and self.total_steps > 10:
            total_hits = self.max_300s_achieved + self.max_100s_achieved + self.max_50s_achieved + self.max_misses
            acc = 0.0
            if total_hits > 0:
                acc = (self.max_300s_achieved * 300 + self.max_100s_achieved * 100 + self.max_50s_achieved * 50) / (total_hits * 300) * 100

            print(f"\n{'='*50}")
            print(f"[MAP SUMMARY]")
            print(f"  Score:    {self.max_score}")
            print(f"  300s:     {self.max_300s_achieved}")
            print(f"  100s:     {self.max_100s_achieved}")
            print(f"  50s:      {self.max_50s_achieved}")
            print(f"  Misses:   {self.max_misses}")
            print(f"  Accuracy: {acc:.2f}%")
            print(f"  Combo:    {self.max_combo_achieved}")
            print(f"  Reward:   {self.current_episode_reward:.4f}")
            print(f"{'='*50}\n")

            obs = self._get_observation()
            return obs, 0.0, True, False, {
                'hits_300': self.max_300s_achieved,
                'hits_100': self.max_100s_achieved,
                'hits_50': self.max_50s_achieved,
                'misses': self.max_misses,
                'combo': self.max_combo_achieved,
                'score': self.max_score,
                'accuracy': acc,
            }

        # ===== WATCHDOG: frozen game =====
        if self.is_playing:
            if self.current_time_ms != self._last_time_ms:
                self._last_time_ms = self.current_time_ms
                self._last_time_wall = time.monotonic()
            elif time.monotonic() - self._last_time_wall > self.STALL_TIMEOUT_S:
                print(f"\n[OsuEnv] WATCHDOG: song time frozen for {self.STALL_TIMEOUT_S:.0f}s "
                      f"— game hung (wine deadlock?). Truncating episode.")
                obs = self._get_observation()
                return obs, 0.0, False, True, {'stalled': True}

        # ===== DECODE ACTION =====
        x_bin, y_bin, click_state = int(action[0]), int(action[1]), int(action[2])

        # ===== EXECUTE MOVEMENT (absolute aim, bin center) =====
        target_x = self.PLAY_LEFT + int((x_bin + 0.5) * self.PLAY_W / self.X_BINS)
        target_y = self.PLAY_TOP + int((y_bin + 0.5) * self.PLAY_H / self.Y_BINS)
        self.pointer.move_to(target_x, target_y)

        # ===== EXECUTE CLICK =====
        if click_state == 1:
            if not self.is_held:
                self.current_key = 'd' if self.current_key == 's' else 's'
                self.keyboard.press(self.current_key)
                self.is_held = True
        else:
            if self.is_held:
                self.keyboard.release(self.current_key)
                self.is_held = False

        # ===== PACE THE LOOP =====
        elapsed = time.monotonic() - self._step_t
        if elapsed < self.STEP_DT:
            time.sleep(self.STEP_DT - elapsed)
        self._step_t = time.monotonic()

        # ===== GET NEW STATE =====
        observation = self._get_observation()
        reward = self._compute_reward()

        self.current_episode_reward += reward
        self.total_steps += 1

        # Periodic logging
        if self.total_steps % 500 == 0:
            now = time.monotonic()
            hz = 500 / max(now - self._rate_t, 1e-6)
            self._rate_t = now
            print(f"[Step {self.total_steps}] Rew: {reward:.4f} | "
                  f"Total: {self.current_episode_reward:.2f} | "
                  f"Miss: {self.current_misses} | "
                  f"Rate: {hz:.1f} steps/s")

        return observation, reward, False, False, {}

    def _clear_search(self):
        """Empty the song-select search box (harmless on other screens)."""
        for _ in range(40):
            self.keyboard.tap('backspace', hold=0.01)
        time.sleep(0.2)

    def select_map(self, query: str, diff_scan: int = 12) -> bool:
        """
        Filter to `query` in song select and select the exact difficulty.

        Two stages, both confirmed via telemetry so nothing fires blindly:

        1. Get the search filter to take effect. Type the query; if the
           selected map changes (or already matches), we're at song select and
           filtering. If typing has no effect, we're on another screen (e.g.
           results) — press Esc once to step toward song select and retry.
           Esc only fires when typing did nothing, so it can't walk out of the
           menus (the failure mode of blind Esc-retry).

        2. Land the right difficulty. osu!'s search filters the beatmap *set*
           but doesn't always select the difficulty whose name you typed, so
           step Down through the filtered carousel until the exact diff (all
           query tokens present in folder+file) is the selected one.

        `query` should carry title + difficulty tokens (e.g. "no title celsius
        easy"). Returns True once the exact map+diff is confirmed selected.
        """
        tokens = query.lower().split()

        def matches():
            # Token-based, like osu!'s own search: every query word must appear
            # in folder+file. Title and difficulty tokens both required, so this
            # only passes for the intended difficulty, not a sibling in the set.
            hay = (self.map_folder + " " + self.map_file).lower()
            return all(tok in hay for tok in tokens)

        print(f"[MapSwitch] Searching for: {query!r}")

        # Stage 1: apply the filter (retry via Esc only if typing had no effect)
        filtered = False
        for _ in range(4):
            before = self.map_file
            self._clear_search()
            self.keyboard.type_text(query)
            time.sleep(1.0)
            if matches() or self.map_file != before:
                filtered = True
                break
            self.keyboard.tap('esc', hold=0.1)  # not at song select — step toward it
            time.sleep(1.2)

        if not filtered:
            print(f"[MapSwitch] WARNING: couldn't filter to {query!r} "
                  f"(current: {self.map_file!r}). Starting whatever is selected.")
            return False

        # Stage 2: step through the filtered set's difficulties to the target
        for _ in range(diff_scan):
            if matches():
                print(f"[MapSwitch] Selected: {self.map_file}")
                return True
            self.keyboard.tap('down', hold=0.05)
            time.sleep(0.6)

        print(f"[MapSwitch] WARNING: filtered but couldn't land the exact diff for "
              f"{query!r} (current: {self.map_file!r}). Starting whatever is selected.")
        return False

    def reset(self, seed=None, options=None):
        """Reset for a new episode."""
        super().reset(seed=seed)

        self.keyboard.release('s')
        self.keyboard.release('d')
        self.is_held = False
        self.current_key = 's'

        if not self.is_playing:
            print("Navigating back to map...")

            # Optional map switch. select_map self-corrects to song select from
            # whatever screen we're on, so no pre-Esc here (a pre-Esc from a
            # fresh song-select start would land in the main menu instead).
            switching = self.pending_map_query is not None
            if switching:
                self.select_map(self.pending_map_query)
                self.pending_map_query = None

            attempts = 0
            while not self.is_playing:
                attempts += 1
                if attempts % 5 == 0:
                    print(f"[OsuEnv] Still can't start a map after {attempts} tries — "
                          f"osu! may be frozen (wine deadlock). If so, restart it: "
                          f"./scripts/start_session.sh")
                # After a switch we're already at song select with the target
                # selected, so the first attempt goes straight to Enter; the
                # no-switch path (and all retries) resync via Esc first.
                if not (switching and attempts == 1):
                    self.keyboard.tap('esc', hold=0.1)
                    time.sleep(1.5)

                self.keyboard.tap('enter', hold=0.1)

                for _ in range(50):
                    time.sleep(0.1)
                    if self.is_playing:
                        break

            time.sleep(1.0)
            self.keyboard.tap('space', hold=0.1)
            print("Map started!")

        # Reset counters
        self.current_score = 0
        self.current_combo = 0
        self.current_misses = 0
        self.current_300s = 0
        self.current_100s = 0
        self.current_50s = 0
        self.prev_misses = 0
        self.prev_300s = 0
        self.prev_100s = 0
        self.prev_50s = 0
        self.prev_combo = 0
        self.max_score = 0
        self.max_combo_achieved = 0
        self.max_misses = 0
        self.max_300s_achieved = 0
        self.max_100s_achieved = 0
        self.max_50s_achieved = 0
        self.current_episode_reward = 0.0
        self.total_steps = 0
        self.episode_shaping = 0.0
        self._step_t = time.monotonic()
        self._rate_t = time.monotonic()
        self._last_time_ms = -1
        self._last_time_wall = time.monotonic()

        self.beatmap.reset()

        # Center mouse
        center_x = (self.PLAY_LEFT + self.PLAY_RIGHT) // 2
        center_y = (self.PLAY_TOP + self.PLAY_BOTTOM) // 2
        self.pointer.move_to(center_x, center_y)

        # Init frame stack
        first_frame = self._get_single_frame()
        for _ in range(self.FRAME_STACK):
            self.frame_stack.append(first_frame)

        return {
            'frames': np.array(self.frame_stack, dtype=np.uint8),
            'state': self._get_game_state(),
        }, {}
