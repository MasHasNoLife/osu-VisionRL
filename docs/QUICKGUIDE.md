# Deeposu 1→100: Zero to Trained Agent

One numbered path from fresh machine to a bot climbing star ratings. Details live in [TRAINING.md](TRAINING.md); this is the checklist you actually follow.

## One-time setup (1–12)

1. Clone the repo and `cd` into it.
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Verify you're in the input group: `groups | grep input` (if not: `sudo usermod -aG input $USER`, re-login).
5. Verify `/dev/uinput` exists and is writable: `ls -la /dev/uinput`.
6. Test virtual input: `python wayland_input.py` — must print `PASS — input backend works` (the desktop cursor may not visibly move; XTest lives inside XWayland, which is where the game looks).
7. Install `v4l2loopback-dkms` and `wf-recorder` from your package manager.
8. Have osu! stable working under wine, and the tosu Windows build (`tosu.exe`) in the repo root.
9. In osu! options: **Raw input OFF**, sensitivity 1.0, fullscreen at native resolution.
10. In osu! options: background dim **100%**, videos/storyboards **off**, keys bound to **S** and **D**.
11. Confirm `tosu.env` has `POLL_RATE=10`.
12. Smoke-test the network: `python agent.py` → should print `~2,530,643 params` and `✓ OK!`.

## Every session: startup (13–20)

13. `./scripts/start_session.sh` — brings up virtual camera, screen feed, tosu (wine), osu! (wine).
14. Wait for `tosu websocket — UP.`
15. Log into osu! (script passes your devserver flag automatically).
16. Press F1 at song select → enable **No Fail**. Non-negotiable for training.
17. Select your current training map (see 31–40).
18. `python train_deeposu.py`
19. Click the osu! window during the 5-second countdown — it must keep focus.
20. Walk away only after one full episode looks healthy (21–30).

## First-run health check (21–30)

21. "Deeposu v3 Vision" window shows the playfield: objects on black, no HUD.
22. Console shows `[KillSwitch] Listening on: <your keyboard>`.
23. The cursor snaps around the playfield (random at first — moving is what matters).
24. Step logs show `Rate: ~60 steps/s`.
25. `Rollout: ... (shaping: X)` with X ≠ 0.00 — if zero, the beatmap didn't load; check for `[BeatmapParser] Loaded`.
26. Episode ends with a `[MAP SUMMARY]` block.
27. `PPO OK. P:... V:... E:...` prints real numbers (any crash here is a genuine bug — report it).
28. `training_report.md` gains a data row.
29. Test the kill switch: press `]` → "KILL SWITCH! Saving..." → restart training → it resumes from the checkpoint.
30. Start TensorBoard in another terminal: `tensorboard --logdir ppo_osu_tensorboard`.

## Choosing maps (31–40)

31. **Phase 1 map:** one Easy diff — CS ≤ 3, AR ≤ 2, mostly circles, 1–2 minutes (e.g. *Sentou! Champion Iris [Noffy's Easy]*).
32. In-game search filter for candidates: `stars<2 cs<3.5 ar<4 length<120`.
33. Prefer: circle-heavy, spread-out patterns, steady rhythm.
34. Avoid (for now): spinner-heavy maps, dense slider chains, 2B/gimmick maps.
35. Short maps beat long maps — more episodes per hour = more PPO updates.
36. Stay on **one** map for all of Phase 1. Overfitting it is the goal.
37. **Phase 2 pool:** 3–5 maps from the same filter, different songs/patterns.
38. **Phase 3 pool:** `stars>2 stars<3.5 ar<6`, still CS ≤ 4.
39. Beyond that, raise stars in +0.5–1.0 increments; never jump more than ~1★ at once.
40. When you switch maps mid-training, just pick the new one at song select between episodes — the env reloads the beatmap automatically.

## Phase 1 — single map (41–55)

41. Target: 300–500 episodes (~1–2 days continuous at ~15 episodes/hour).
42. **Phase 1a: enable RX + NF mods** — Relax auto-clicks, so hits directly reward aim; train aim first, tapping later.
43. Episodes 1–20: `rollout/shaping_reward` **rises** from ~205 toward the ~315 ceiling (summed proximity bonus; higher = more cursor time on target; ~205 = random).
44. Flat at ~21 by episode 20 → check `train/epochs_completed` (pinned at 1 = `TARGET_KL` too tight) and `train/entropy` (stuck at 8.7 = `ENTROPY_COEF` too strong).
45. Episodes 20–100: RX accuracy climbs steeply as aim solidifies.
46. **Phase 1b: drop RX (keep NF)** once RX accuracy ~80% — accuracy dips, then click timing grinds up through real hits, by design.
47. Watch `train/entropy`: starts ~8.72, should decay visibly within tens of episodes; < 3.0 while accuracy is still low → raise `ENTROPY_COEF`.
48. Watch `train/epochs_completed`: healthy is mostly 2–4 with occasional KL early stops.
49. KL early-stops at epoch 1 every episode → halve `LR` to 1e-4.
50. Overnight runs are safe: No Fail keeps episodes uniform, checkpoints save every episode.
51. Glance at a stuck "Navigating back to map..." after long unattended runs — a stray dialog can trap menu navigation; press `]` and restart if so.
52. Don't touch mouse/keyboard while training — the bot owns the focused window.
53. Disable screen lock/blanking — a locked screen = black frames = blind agent.
54. Best checkpoint (`ppo_v3_best.pth`) and latest (`ppo_v3_latest.pth`) live in `models/Deeposu_v3/`.
55. **Graduate when:** accuracy consistently > 50–60% and misses roughly halved.

## Phase 2 — map pool (56–70)

56. Add the 3–5 pool maps (from 37).
57. Rotate: ~50 episodes per map, round-robin.
58. Expect an accuracy dip on every new map — normal.
59. The dip should shrink with each rotation cycle; that shrinkage *is* generalization.
60. Keep TensorBoard open; compare `osu/accuracy` across rotations, not within one map.
61. If one map lags badly, give it extra rotations rather than dropping it.
62. Backup your models dir occasionally: `cp -r models/Deeposu_v3 models/Deeposu_v3.bak-$(date +%F)`.
63. Keep episode notes (map + episode range) in a scratch file — future-you will want them.
64. Target: 500–1000 episodes (~a week of nights).
65. Mid-phase sanity: `rollout/shaping_reward` should stay high on *every* map — if it drops on one map only, its patterns are the problem (sliders? spacing?).
66. Small slider-heavy map struggling? That's expected — slider *bodies* aren't shaped yet (roadmap item).
67. Value loss spiking on map switches is normal; it re-estimates returns for new patterns.
68. Resist hyperparameter churn: change one knob at a time, give it 50+ episodes.
69. Commit code changes to git as you experiment — checkpoints aren't reproducible without the code that made them.
70. **Graduate when:** ~70%+ accuracy across the whole pool.

## Phase 3 — climbing stars (71–85)

71. New pool: 2–3.5★, AR 4–6 (filter from 38). Same rotation discipline.
72. No action-space changes needed — absolute aim has no speed cap.
73. Jumps now matter: watch the agent learn to pre-position during approach circles.
74. 3–4★: stream sections appear; click alternation (S↔D) is automatic, timing is the skill.
75. Expect longer plateaus per star bracket — each +1★ roughly doubles required precision.
76. Keep one "benchmark map" per bracket; re-run it every ~100 episodes to measure real progress.
77. From ~4★, consider capture at higher FPS if reaction seems late (wf-recorder follows compositor rate).
78. From ~5★ (OD 8–9), the 60 Hz click granularity (±16.7ms) starts costing 300s — that's the sub-step click-timing roadmap item, not a tuning problem.
79. AR 9+ gives ~450–600ms object visibility — the 4-frame stack still covers it at 60 Hz, but the game-state vector (which sees 2s ahead) becomes the agent's main early warning.
80. Slider breaks (`-0.3` penalty) become the dominant loss at 5★+; slider-path shaping is the next big win.
81. Spinners: currently unshaped and mostly ignored — cheap accuracy loss until spinner-aware targets land.
82. If progress stalls a full bracket, drop back 0.5★ for 100 episodes (curriculum breathing room).
83. Log milestone results into README's results section — that's your portfolio evidence.
84. Record a gameplay clip per bracket (wf-recorder is already running!) for the README.
85. 7–8★ is a research goal, not a checklist item: expect months of accumulated training plus the roadmap items (sub-step clicks, slider shaping, higher-rate capture).

## Troubleshooting (86–95)

86. Cursor doesn't move → `python wayland_input.py` square test; check `/dev/uinput` perms and input group.
87. Cursor moves on desktop but not in osu! → Raw input snuck back ON in osu! options; turn it off.
88. Shaping always 0.00 → tosu not connected (port 24050) or beatmap path mismatch; check `logs/tosu-runtime.log`.
89. Vision window black → wf-recorder died or screen locked; re-run `./scripts/start_session.sh`.
90. Vision shows menus during gameplay → wrong `/dev/video9` source; check nothing else grabbed the loopback.
91. `Rate:` far below 60 → capture bottleneck; check wf-recorder CPU usage and compositor frame rate.
92. Misses counting while it plays fine → tosu offsets broke after an osu! update; update tosu.
93. Keys not registering → osu! keys rebound away from S/D, or a non-us layout became active.
94. `]` kill switch dead → listener found no keyboard; verify input group (step 4) and re-login.
95. Training crash mid-episode → the traceback is real (nothing is swallowed); file it, restart, checkpoints resume.

## Long game (96–100)

96. Re-verify the offline test after any code change to agent/ppo: it must still pass before wasting a live session.
97. Track wall-clock training time; RL progress correlates with steps collected, not calendar days.
98. When results land, update the README roadmap checkboxes and results table — keep the repo CV-ready.
99. Keep bot scores off official leaderboards; private/local play only.
100. When the bot out-aims you, congratulations — open the "sub-step click timing" roadmap item and start the climb to 7–8★.
