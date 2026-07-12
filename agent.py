"""
Agent architecture v3 — scaled for high-difficulty play.

- Nature-CNN visual backbone (Mnih et al. 2015) over 4 stacked 96x96 frames
- 8-dim game-state vector (cursor position, next-object geometry from
  tosu + beatmap parsing) fused into the trunk
- Factorized discrete action heads: absolute aim X (64 bins) x Y (48 bins)
  x click (2 states). Absolute positioning has no cursor speed cap, so the
  same action space scales from 1-star to 8-star maps; bin granularity
  (~28px on a 2560x1440 playfield) is well inside even a CS7 circle (~82px).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-critic with factorized absolute-aim action space."""

    X_BINS = 64
    Y_BINS = 48
    CLICK_STATES = 2
    STATE_DIM = 8

    def __init__(self):
        super().__init__()

        # ===== VISION: Nature CNN =====
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),   # (4,96,96) -> (32,23,23)
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # -> (64,10,10)
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # -> (64,8,8)
            nn.ReLU(),
            nn.Flatten(),                                # -> 4096
            layer_init(nn.Linear(4096, 512)),
            nn.ReLU(),
        )

        # ===== GAME STATE =====
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(self.STATE_DIM, 64)),
            nn.ReLU(),
        )

        # ===== FUSION TRUNK =====
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(512 + 64, 512)),
            nn.ReLU(),
        )

        # ===== HEADS =====
        self.actor_x = layer_init(nn.Linear(512, self.X_BINS), std=0.01)
        self.actor_y = layer_init(nn.Linear(512, self.Y_BINS), std=0.01)
        self.actor_click = layer_init(nn.Linear(512, self.CLICK_STATES), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def _features(self, frames, state):
        """frames: (B,4,96,96) float in [0,1]; state: (B,8) float."""
        v = self.cnn(frames)
        s = self.state_fc(state)
        return self.trunk(torch.cat([v, s], dim=-1))

    def get_action_and_value(self, frames, state, action=None):
        """
        action: optional (B,3) long tensor [x_bin, y_bin, click] for training.
        Returns summed log_prob/entropy over the three factorized heads.
        """
        h = self._features(frames, state)

        dist_x = Categorical(logits=self.actor_x(h))
        dist_y = Categorical(logits=self.actor_y(h))
        dist_c = Categorical(logits=self.actor_click(h))

        if action is None:
            action = torch.stack(
                [dist_x.sample(), dist_y.sample(), dist_c.sample()], dim=-1
            )

        log_prob = (dist_x.log_prob(action[:, 0]) +
                    dist_y.log_prob(action[:, 1]) +
                    dist_c.log_prob(action[:, 2]))
        entropy = dist_x.entropy() + dist_y.entropy() + dist_c.entropy()
        value = self.critic(h)

        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value,
        }

    def get_value(self, frames, state):
        return self.critic(self._features(frames, state))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent().to(device)
    params = sum(p.numel() for p in agent.parameters())
    print(f"Params: {params:,} | Device: {device}")

    with torch.no_grad():
        frames = torch.rand(2, 4, 96, 96, device=device)
        state = torch.rand(2, 8, device=device)
        r = agent.get_action_and_value(frames, state)
        print(f"Action: {r['action'].tolist()}")
        print(f"Entropy: {r['entropy'][0].item():.4f} (max = ln64+ln48+ln2 = "
              f"{np.log(64)+np.log(48)+np.log(2):.4f})")
        print(f"Value: {r['value'][0].item():.4f}")
        # evaluate mode with given actions
        r2 = agent.get_action_and_value(frames, state, action=r['action'])
        assert torch.allclose(r['log_prob'], r2['log_prob'])
        print("✓ OK!")
