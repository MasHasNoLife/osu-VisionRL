"""
PPO Trainer for Discrete Action Space.
Clean implementation with proper GAE returns and mixed precision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os


def calculate_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GAE)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        next_value = next_value * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """PPO trainer for discrete action space."""
    
    def __init__(self,
                 agent,
                 lr=3e-4,
                 ppo_clip=0.2,
                 target_kl=0.02,
                 train_epochs=10,
                 num_minibatches=8,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 checkpoint_dir="./models/Deeposu_v3/"):

        self.agent = agent
        self.ppo_clip = ppo_clip
        self.target_kl = target_kl
        self.train_epochs = train_epochs
        self.num_minibatches = num_minibatches
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_best = os.path.join(checkpoint_dir, "ppo_v3_best.pth")
        self.checkpoint_latest = os.path.join(checkpoint_dir, "ppo_v3_latest.pth")
        
        self.optimizer = optim.AdamW(agent.parameters(), lr=lr, eps=1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
    
    def train(self, rollout_data):
        """Train PPO on a collected rollout.

        rollout_data['frames'] is uint8 (T, 4, 96, 96) — normalized to [0,1]
        per-minibatch on the GPU to keep host RAM usage 4x lower.
        rollout_data['states'] is float32 (T, 8).
        rollout_data['actions'] is long (T, 3) — factorized [x_bin, y_bin, click].
        """
        device = next(self.agent.parameters()).device

        frames = rollout_data['frames']
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['old_log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']

        T = len(frames)
        minibatch_size = max(1, T // self.num_minibatches)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_kl_divs = []
        all_clipfracs = []
        epochs_done = 0

        for epoch in range(self.train_epochs):
            indices = torch.randperm(T)

            for mb_start in range(0, T, minibatch_size):
                mb_end = min(mb_start + minibatch_size, T)
                mb_idx = indices[mb_start:mb_end]

                mb_frames = frames[mb_idx].to(device).float() / 255.0
                mb_states = states[mb_idx].to(device)
                mb_acts = actions[mb_idx].to(device)
                mb_old_lp = old_log_probs[mb_idx].to(device)
                mb_adv = advantages[mb_idx].to(device)
                mb_ret = returns[mb_idx].to(device)

                with torch.amp.autocast('cuda'):
                    result = self.agent.get_action_and_value(mb_frames, mb_states, action=mb_acts)

                    new_lp = result['log_prob']
                    entropy = result['entropy']
                    values = result['value'].squeeze(-1)

                    log_ratio = new_lp - mb_old_lp
                    ratio = log_ratio.exp()

                    surr1 = -mb_adv * ratio
                    surr2 = -mb_adv * torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                    policy_loss = torch.max(surr1, surr2).mean()

                    value_loss = 0.5 * ((mb_ret - values) ** 2).mean()
                    entropy_loss = entropy.mean()

                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy_loss.item())

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    all_kl_divs.append(approx_kl)
                    all_clipfracs.append(
                        ((ratio - 1.0).abs() > self.ppo_clip).float().mean().item())

            epochs_done = epoch + 1
            avg_kl = np.mean(all_kl_divs[-self.num_minibatches:])
            if avg_kl > self.target_kl:
                print(f"  Early stopping at epoch {epoch+1}/{self.train_epochs} (KL={avg_kl:.4f})")
                break

        stats = {
            'policy_loss': float(np.mean(all_policy_losses)),
            'value_loss': float(np.mean(all_value_losses)),
            'entropy': float(np.mean(all_entropies)),
            'approx_kl': float(np.mean(all_kl_divs[-self.num_minibatches:])),
            'clipfrac': float(np.mean(all_clipfracs)),
            'epochs': epochs_done,
        }
        return True, stats
    
    def save_checkpoint(self, path, best_score=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        if best_score is not None:
            checkpoint['best_score'] = best_score
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint.get('best_score', None)
    
    def save_best(self, best_score):
        self.save_checkpoint(self.checkpoint_best, best_score)
    
    def save_latest(self, best_score):
        self.save_checkpoint(self.checkpoint_latest, best_score)
