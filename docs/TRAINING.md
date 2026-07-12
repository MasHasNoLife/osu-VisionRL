# Training Runbook

The complete operational guide for training Deeposu: session setup, game configuration, the difficulty curriculum, expected learning timeline, and how to interpret every monitoring signal.

## 1. One-time osu! configuration

| Setting | Value | Why |
|---|---|---|
| Raw input | **OFF** | The uinput pointer is absolute; the game must follow the OS cursor 1:1 |
| Mouse sensitivity | 1.0 | Any scaling breaks the screen-coordinate mapping |
| Keyboard keys | S and D | These are the keys the environment presses |
| Background dim | 100% | Clean vision — objects on black |
| Video / storyboards | OFF | Removes visual noise the CNN would have to learn to ignore |
| **No Fail mod (NF)** | **ON** | Without it the agent fails early, never sees most of the map, and episode lengths vary wildly. With NF every episode covers the full map. |

Also disable screen locking/blanking (a locked screen feeds black frames — the agent goes blind), and don't use the PC during training: the virtual keyboard types into whatever window has focus.

## 2. Session startup (every time)

One command brings up the whole stack (idempotent — re-run any time):

```bash
./scripts/start_session.sh
```

Then select the training map in osu! and run `python train_deeposu.py`, focusing the osu! window during the 5s countdown.

<details>
<summary>Manual equivalent (what the script does)</summary>

```bash
# 1. Virtual camera
sudo modprobe v4l2loopback video_nr=9 card_label="VirtualCam" exclusive_caps=1

# 2. Screen -> /dev/video9 (leave running)
wf-recorder -c rawvideo -m v4l2 -x yuv420p -f /dev/video9

# 3. tosu — the Windows build must run in the SAME wine prefix as osu!,
#    so ReadProcessMemory can reach the osu! process through the shared
#    wineserver. Run from the repo dir so it reads tosu.env (POLL_RATE=10).
WINEPREFIX=~/.osu-wine wine ./tosu.exe

# 4. osu! itself
WINEPREFIX=~/.osu-wine wine ~/.osu-wine/drive_c/users/$USER/AppData/Local/'osu!'/'osu!.exe'
```

</details>

### Wine notes

- **`RawInput = 0` is required** in `osu!.<user>.cfg`. With raw input on, wine
  synthesizes Windows raw input from XInput2 — fragile with absolute devices.
  With it off, osu! follows the OS cursor, which the uinput absolute pointer
  sets to exact pixels: a deterministic 1:1 mapping. (Same reason tablet
  players are fine either way — but the bot needs determinism.)
- Input uses the **XTest backend by default** (`DEEPOSU_INPUT=xtest`): absolute
  warps injected directly into the XWayland server wine runs on, verified by
  position readback. The kernel-uinput backend (`DEEPOSU_INPUT=uinput`) moves
  the compositor cursor but on Hyprland the motion never reaches XWayland
  clients — the game cursor stays frozen. Note: with XTest, the *desktop*
  cursor may not visibly move; the in-game osu! cursor is what follows.
- Keyboard events map through the XKB layout: a plain `us` layout is assumed
  for the S/D virtual key presses.
- A `-devserver` flag on the osu! command does not affect tosu (it reads
  process memory, not network traffic).

**Kill switch:** press `]` on the physical keyboard — saves `ppo_v2_latest.pth` and exits cleanly. Checkpoints are also written after every episode, so a crash never costs more than one episode.

## 3. Curriculum — what to train on

### Phase 0 — smoke test (10 minutes)

Run one or two episodes and verify, in order:

1. Console prints `Input: uinput absolute pointer + virtual keyboard` and `[KillSwitch] Listening on: <your keyboard>`.
2. The cursor visibly moves (jittery random motion is expected at first).
3. `Rollout: N steps, reward: X (shaping: Y)` — **shaping must be nonzero**. If exactly 0.00: the beatmap didn't load (look for `[BeatmapParser] Loaded ...`) or tosu isn't connected.
4. After the episode: `PPO OK. P:... V:... E:...` — the trainer fails loudly by design; any crash is a real bug.
5. `training_report.md` gains a row with real loss numbers.
6. Test the `]` kill switch once.

### Phase 1 — single-map mastery (~300–500 episodes, 1–2 days continuous)

Use one short, low-difficulty map — large circles (CS ≤ 3), slow approach (AR ≤ 2), mostly circles, 1–2 minutes long. Do **not** switch maps: overfitting one map is the goal of this phase; it proves the learning loop before spending compute on generalization.

Each episode ≈ map length + 30–60 s of PPO update ≈ 3.5–4.5 min → roughly 14–17 episodes/hour.

**Phase 1a — aim only, with Relax (RX + NF mods).** RX auto-clicks, so every
correctly-aimed object scores — the hit reward becomes dense, direct feedback
on aim instead of requiring aim × click-timing to coincide by luck. The click
head receives no meaningful gradient during RX (its actions are ignored),
which is fine: aim knowledge lives in the shared trunk and X/Y heads.

**Phase 1b — add tapping (drop RX, keep NF)** once RX accuracy is high
(~80%+). Aim carries over; the click head learns timing on top of it.

| Episodes (1a) | Expected behavior |
|---|---|
| 1–20 | `rollout/shaping_reward` (per-step proximity bonus, summed) **rises** from ~205 (random aim) toward the ~315 ceiling as cursor hover-time on target grows. Cursor visibly favors objects. |
| 20–100 | RX accuracy climbs steeply — with aim ≈ solved per-object, hits follow directly. |
| 1b onward | Accuracy dips when RX comes off (timing is now real), then grinds back up through actual hits — click timing is deliberately unshaped (shaping clicks would teach click-spam). |

If shaping is *not* trending down within ~20 episodes, check `train/epochs_completed`
(pinned at 1 = KL cap too tight) and `train/entropy` (stuck at 8.7 = entropy
bonus too strong), and confirm the state vector isn't all zeros (beatmap must be loaded).

**Graduate when:** accuracy consistently >50–60% and misses are roughly half of what an untrained agent gets.

### Phase 2 — generalization (~500–1000 episodes, about a week)

Pick 3–5 more maps (in-game filter: `stars<2 cs<3.5 ar<4 length<120`). Prioritize short, circle-heavy, low-AR maps.

**Automatic rotation (recommended).** Fill `MAP_POOL` in `train_deeposu.py` with a distinctive search substring per map — exactly what you'd type in osu!'s song-select search to land on that map (e.g. `"champion iris"`, `"miiro normal"`). The trainer rotates round-robin every `ROTATE_EVERY` (default 50) episodes: it types the query into the search box, confirms the selection via telemetry, and starts it. Keep the queries unique enough to land one map. Requires the default XTest input backend. Leave `MAP_POOL = []` to disable and rotate manually.

**Manual rotation.** Switch the selected map at song select yourself — the environment detects the change through telemetry and reloads the beatmap automatically (watch for the `[BeatmapParser] Loaded ...` and `[Map] Now training on:` lines).

Accuracy temporarily drops with each new map; it recovers faster each time as the policy generalizes.

### Phase 3 — difficulty scaling (open-ended)

Once the pool plays at ~70%+ accuracy, move to 2–3★, AR 4–6, then upward. Absolute aim has no cursor speed cap, so no action-space changes are needed for jump patterns. The practical ceilings at high star ratings are click-timing granularity (~16ms at 60Hz vs ±19.5ms 300-windows at OD9) and AR9+ reaction time — see the roadmap items for sub-step click timing and higher-rate capture when you get there.

## 4. Monitoring

`tensorboard --logdir ppo_osu_tensorboard`

| Curve | Healthy | Warning sign → fix |
|---|---|---|
| `rollout/shaping_reward` | **Rises** from ~205 (random aim) toward ~315 (perfect hover). It's the summed per-step proximity bonus: 0.05 × (1 − distance) per step, so higher = more time on target. | Flat at ~205 for 30+ eps → aim not learning; check `epochs_completed` and entropy. Exactly 0.00 → beatmap/tosu not connected |
| `rollout/hit_reward` | Climbs steadily (fast under RX; slower in 1b) | Shaping low (aim good) but hits flat after 300+ eps of 1b → timing bottleneck; consider a click-timing signal |
| `train/entropy` | Starts ~8.72 (ln64+ln48+ln2), decays visibly within tens of episodes | <3.0 while accuracy still low = premature collapse → raise `ENTROPY_COEF`. Stuck ~8.7 after 50 eps → lower `ENTROPY_COEF` further / raise `TARGET_KL` |
| `train/epochs_completed`, `train/approx_kl` | Mostly 2–4 epochs; occasional early stops | Pinned at 1 every episode → `TARGET_KL` too tight for summed-head KL, raise it (or lower `LR`) |
| `osu/accuracy`, `osu/misses` | Ground truth — judge progress here | — |

## 5. Expectations & practical notes

- Pure RL from pixels is the slow road (that's the price of zero supervised data). Budget **~2 weeks of accumulated training** for respectable easy-map play. First visible learning (shaping curve) within the first hour; first satisfying gameplay within 1–2 days.
- Overnight runs are safe: NF keeps episodes uniform, the env restarts the map indefinitely, every episode checkpoints.
- Known rough edge: between episodes the env navigates menus blindly with Esc/Enter. If osu! lands somewhere unexpected (dialog, overlay), an episode can hang at "Navigating back to map...". Glance at it every few hours; if stuck, `]` and restart — nothing is lost.
