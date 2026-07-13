"""
Training loop for osu! AI v3.
Absolute-aim factorized actions + Nature CNN + game-state fusion + GAE.
"""

import os
import re
import time
from collections import deque, defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from osu_env import OsuEnv
from agent import Agent
from ppo import PPOTrainer, calculate_gae
from wayland_input import KillSwitchListener


def map_label(map_file: str) -> str:
    """Short, stable label for a beatmap: 'Song [Diff]' -> tag for grouping.

    TensorBoard splits scalars on '/', so we sanitize to keep one clean tag
    per map. Falls back to 'unknown' before the first beatmap loads.
    """
    if not map_file:
        return "unknown"
    stem = map_file[:-4] if map_file.lower().endswith(".osu") else map_file
    diff = re.search(r"\[([^\]]+)\]\s*$", stem)  # difficulty name in [ ]
    if diff:
        title = stem[:diff.start()].strip(" -")
        artist_title = title.split(" - ")[-1].strip()  # drop 'Artist - '
        label = f"{artist_title} [{diff.group(1)}]"
    else:
        label = stem
    return re.sub(r"[^\w \[\]\-]", "", label)[:60].strip() or "unknown"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GAMES = 100_000
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 2.5e-4         # Atari-PPO default; KL early stop guards against blowups
PPO_CLIP = 0.1
# Entropy/KL are summed over 3 factorized heads (~3x a single head's scale),
# so both are rescaled vs single-head defaults. 0.01/0.02 pinned the policy
# at uniform: entropy never left 8.7 over 15 episodes.
ENTROPY_COEF = 0.003
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TRAIN_EPOCHS = 6    # KL sits ~0.015 of the 0.06 cap — room for more reuse
NUM_MINIBATCHES = 8
TARGET_KL = 0.06
CHECKPOINT_DIR = "./models/Deeposu_v3/"

# ===== MAP ROTATION =====
# Each entry is a distinctive search substring of a beatmap's title/difficulty —
# what you'd type in osu!'s song-select search to land on exactly that map.
# The trainer rotates through the pool every ROTATE_EVERY episodes (round-robin).
# Leave MAP_POOL empty ([]) to disable rotation and train the currently-selected
# map only (original behavior). Requires the XTest input backend (the default).
# One map per run (50 episodes), so you can break between maps. Uncomment ONE
# line at a time and re-run; move down the list each session. Weights carry
# over via the checkpoint.
#
# PHASE 1b (current): RX OFF in osu! (keep NF) — real click-timing training.
# RX revisit cycle completed 2026-07-10 at ~50-60% aim on all four maps;
# checkpoint backed up to "model backup/5 (rx cycle complete)".
MAP_POOL = [
    # "champion iris noffy easy",     # Champion Iris — timing ~26% after 60 eps ✓
    # "no title celsius easy",        # No title — timing ~22-25% after 50 eps ✓
    # "zen zen zense music box normal", # Zen Zen Zense — timing ~22-23% after 50 eps ✓
    "ninja ryuu easy",                # PLight - NINJA [ryuu's Easy] — timing next (last of cycle)
]
ROTATE_EVERY = 50
# Stop after one full pass through the pool (len(MAP_POOL) * ROTATE_EVERY
# episodes), then save and exit. Set STOP_AFTER_ONE_CYCLE = False to loop the
# pool forever (revisiting maps — the generalization curriculum).
STOP_AFTER_ONE_CYCLE = True


kill_switch_activated = False
def activate_kill_switch():
    global kill_switch_activated
    kill_switch_activated = True


def main():
    global kill_switch_activated
    
    print("=" * 60)
    print("  Deeposu v3 — Absolute-Aim PPO")
    print("  Nature CNN + game-state fusion + factorized actions")
    print("=" * 60)
    
    env = OsuEnv()
    agent = Agent().to(DEVICE)
    ppo = PPOTrainer(
        agent, lr=LR, ppo_clip=PPO_CLIP, target_kl=TARGET_KL,
        train_epochs=TRAIN_EPOCHS, num_minibatches=NUM_MINIBATCHES,
        entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM, checkpoint_dir=CHECKPOINT_DIR
    )
    
    best_score = -float('inf')
    ckpt = None
    if os.path.exists(ppo.checkpoint_latest):
        ckpt = ppo.checkpoint_latest
    elif os.path.exists(ppo.checkpoint_best):
        ckpt = ppo.checkpoint_best
    
    if ckpt:
        print(f"\nLoading: {os.path.basename(ckpt)}")
        try:
            loaded = ppo.load_checkpoint(ckpt)
            if loaded is not None:
                best_score = loaded
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("\nFresh start!")
    
    writer = SummaryWriter('ppo_osu_tensorboard/Deeposu_v3')

    report_path = "training_report.md"
    if not os.path.exists(report_path):
        with open(report_path, "w") as f:
            f.write("# Deeposu v3 Training Log\n\n")
            f.write("| Ep | Map | Reward | Miss | 300 | 100 | 50 | Acc% | PLoss | VLoss | Ent | Steps |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    # Per-map accuracy history (rolling) so revisit-recovery is visible at a glance
    per_map_acc = defaultdict(lambda: deque(maxlen=10))
    prev_map_label = None
    
    # Kill switch: press ']' on the physical keyboard (evdev — works on Wayland)
    listener = KillSwitchListener(activate_kill_switch).start()
    
    # Bound the run to one full pool cycle if requested.
    if MAP_POOL and STOP_AFTER_ONE_CYCLE:
        total_episodes = len(MAP_POOL) * ROTATE_EVERY
        print(f"\nRun length: {total_episodes} episodes "
              f"({len(MAP_POOL)} maps x {ROTATE_EVERY}), then save + exit.")
    else:
        total_episodes = NUM_GAMES

    print(f"\nParams: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for episode in range(total_episodes):
        print(f"\n--- Episode {episode} ---")

        # Map rotation: at each ROTATE_EVERY boundary, queue the next pool map.
        # env.reset() performs the actual in-game switch before starting.
        if MAP_POOL and episode % ROTATE_EVERY == 0:
            next_map = MAP_POOL[(episode // ROTATE_EVERY) % len(MAP_POOL)]
            env.pending_map_query = next_map
            print(f"[MapSwitch] Rotating to pool map: {next_map!r}")

        obs, _ = env.reset()
        cur_map = map_label(env.map_file)
        if cur_map != prev_map_label:
            print(f"[Map] Now training on: {cur_map}")
            writer.add_text('map/switch', cur_map, episode)
            prev_map_label = cur_map

        frames_t = torch.from_numpy(obs['frames']).to(DEVICE).float().unsqueeze(0) / 255.0
        state_t = torch.from_numpy(obs['state']).to(DEVICE).unsqueeze(0)

        r_frames = []
        r_states = []
        r_acts = []
        r_lps = []
        r_rews = []
        r_vals = []
        r_dones = []

        ep_reward = 0.0
        n_steps = 0
        done = False
        stalled = False

        agent.eval()

        while not done:
            if kill_switch_activated:
                print("\nKILL SWITCH! Saving...")
                ppo.save_latest(best_score)
                writer.close()
                return

            with torch.no_grad():
                result = agent.get_action_and_value(frames_t, state_t)

            action = result['action']       # (1,3) [x_bin, y_bin, click]
            log_prob = result['log_prob']
            value = result['value']

            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated
            stalled = stalled or info.get('stalled', False)

            r_frames.append(torch.from_numpy(obs['frames']))  # uint8 — 4x less RAM
            r_states.append(torch.from_numpy(obs['state']))
            r_acts.append(action.squeeze(0).cpu())
            r_lps.append(log_prob.cpu())
            r_rews.append(reward)
            r_vals.append(value.squeeze().item())
            r_dones.append(float(done))

            obs = next_obs
            frames_t = torch.from_numpy(obs['frames']).to(DEVICE).float().unsqueeze(0) / 255.0
            state_t = torch.from_numpy(obs['state']).to(DEVICE).unsqueeze(0)

            ep_reward += reward
            n_steps += 1

            if not env.is_playing and n_steps > 10:
                done = True

        if stalled:
            print("Stalled episode (frozen game) — discarding rollout, not training on it.")
            del r_frames, r_states, r_acts, r_lps, r_rews, r_vals, r_dones
            torch.cuda.empty_cache()
            continue

        # GAE
        rewards = np.array(r_rews, dtype=np.float32)
        values = np.array(r_vals, dtype=np.float32)
        dones = np.array(r_dones, dtype=np.float32)
        advantages, returns = calculate_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
        
        rollout_data = {
            'frames': torch.stack(r_frames),   # uint8, normalized inside ppo.train
            'states': torch.stack(r_states),   # float32 (T,8)
            'actions': torch.stack(r_acts),    # long (T,3)
            'old_log_probs': torch.cat(r_lps),
            'advantages': torch.tensor(advantages),
            'returns': torch.tensor(returns),
        }

        print(f"Rollout: {n_steps} steps, reward: {ep_reward:.2f} "
              f"(shaping: {env.episode_shaping:.2f})")
        
        agent.train()
        success, st = ppo.train(rollout_data)

        if success:
            print(f"PPO OK. P:{st['policy_loss']:.4f} V:{st['value_loss']:.4f} "
                  f"E:{st['entropy']:.4f} KL:{st['approx_kl']:.4f} "
                  f"clip:{st['clipfrac']:.3f} epochs:{st['epochs']}/{TRAIN_EPOCHS}")

        # Logging
        writer.add_scalar('rollout/reward', ep_reward, episode)
        writer.add_scalar('rollout/shaping_reward', env.episode_shaping, episode)
        writer.add_scalar('rollout/hit_reward', ep_reward - env.episode_shaping, episode)
        writer.add_scalar('rollout/steps', n_steps, episode)
        if success:
            writer.add_scalar('train/policy_loss', st['policy_loss'], episode)
            writer.add_scalar('train/value_loss', st['value_loss'], episode)
            writer.add_scalar('train/entropy', st['entropy'], episode)
            writer.add_scalar('train/approx_kl', st['approx_kl'], episode)
            writer.add_scalar('train/clipfrac', st['clipfrac'], episode)
            writer.add_scalar('train/epochs_completed', st['epochs'], episode)
        
        total_hits = env.max_300s_achieved + env.max_100s_achieved + env.max_50s_achieved
        acc = 0.0
        if total_hits + env.max_misses > 0:
            acc = ((env.max_300s_achieved * 300) + (env.max_100s_achieved * 100) + (env.max_50s_achieved * 50)) / ((total_hits + env.max_misses) * 300) * 100
        
        writer.add_scalar('osu/accuracy', acc, episode)
        writer.add_scalar('osu/misses', env.max_misses, episode)
        writer.add_scalar('osu/300s', env.max_300s_achieved, episode)

        # Per-map accuracy: separate TB curve per map + rolling avg for console
        writer.add_scalar(f'per_map_accuracy/{cur_map}', acc, episode)
        per_map_acc[cur_map].append(acc)
        rolling = np.mean(per_map_acc[cur_map])
        print(f"Acc: {acc:.1f}%  |  {cur_map} rolling10: {rolling:.1f}% "
              f"({len(per_map_acc[cur_map])} eps)")
        
        if ep_reward > best_score:
            print(f"New best! {ep_reward:.2f} > {best_score:.2f}")
            best_score = ep_reward
            ppo.save_best(best_score)
        ppo.save_latest(best_score)
        
        with open(report_path, "a") as f:
            pl = f"{st['policy_loss']:.4f}" if success else "N/A"
            vl = f"{st['value_loss']:.4f}" if success else "N/A"
            en = f"{st['entropy']:.4f}" if success else "N/A"
            f.write(f"| {episode} | {cur_map} | {ep_reward:.2f} | {env.max_misses} | "
                    f"{env.max_300s_achieved} | {env.max_100s_achieved} | {env.max_50s_achieved} | "
                    f"{acc:.1f} | {pl} | {vl} | {en} | {n_steps} |\n")
        
        del rollout_data, r_frames, r_states, r_acts, r_lps, r_rews, r_vals, r_dones
        torch.cuda.empty_cache()
        
        if episode % 5 == 0:
            print(f"[VRAM] {torch.cuda.memory_allocated()/1024/1024:.0f}MB")

    # Normal completion: save final progress and exit cleanly.
    print(f"\n{'='*60}")
    print(f"  Completed {total_episodes} episodes — one full pool cycle.")
    ppo.save_latest(best_score)
    print(f"  Saved: {ppo.checkpoint_latest}")
    print(f"  Best reward this run tracked in: {ppo.checkpoint_best}")
    print(f"{'='*60}")
    writer.close()


if __name__ == "__main__":
    main()
