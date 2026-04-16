"""
Deep Q-Network (DQN) agent.

Architecture
────────────
  • 3-layer MLP  (state_dim → 128 → 128 → n_actions)
  • Experience replay buffer (FIFO deque)
  • Hard target-network sync every `target_update` optimiser steps
  • Gradient clipping (max norm 1.0) for stable training
  • ε-greedy exploration with multiplicative decay

Interface is intentionally compatible with BaseAgent so that the
same train_agent / run_episode_visual helpers work unchanged.
"""

import numpy as np
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Neural network ────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """3-layer fully-connected Q-network."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent compatible with the tabular-agent interface used in train.py.

    Parameters
    ----------
    state_dim      : dimensionality of the state vector (default 8)
    n_actions      : number of discrete actions (default 3)
    alpha          : Adam learning rate
    gamma          : discount factor
    epsilon        : initial exploration rate
    epsilon_min    : minimum exploration rate
    epsilon_decay  : multiplicative decay applied after every step
    buffer_size    : maximum replay-buffer capacity
    batch_size     : mini-batch size for each gradient update
    target_update  : sync target network every N optimiser steps
    """

    name  = "DQN"
    color = "#ec4899"   # pink — distinct from the three tabular agents

    def __init__(
        self,
        state_dim: int       = 8,
        n_actions: int       = 3,
        alpha: float         = 0.001,
        gamma: float         = 0.95,
        epsilon: float       = 1.0,
        epsilon_min: float   = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int     = 10_000,
        batch_size: int      = 64,
        target_update: int   = 200,
        train_freq: int      = 4,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQNAgent.\n"
                "Install it with:  pip install torch"
            )

        self.state_dim      = state_dim
        self.n_actions      = n_actions
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.target_update  = target_update
        self.train_freq     = train_freq
        self._opt_steps     = 0          # counts gradient updates
        self._total_steps   = 0          # counts environment steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory    = deque(maxlen=buffer_size)

    # ── Public interface (mirrors BaseAgent) ──────────────────────────────────

    def choose_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.policy_net(t).argmax(dim=1).item())

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """Store transition and perform learning step at fixed intervals."""
        self.memory.append(
            (state, action, reward, next_state, float(done))
        )
        self._total_steps += 1
        
        # Train every N environment steps to speed up execution
        if len(self.memory) >= self.batch_size and self._total_steps % self.train_freq == 0:
            self._learn()
            
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self) -> int:
        """Returns replay buffer occupancy."""
        return len(self.memory)

    # ── Q-value sample (for distribution chart in app.py) ────────────────────

    def sample_q_values(self, n: int = 500) -> np.ndarray:
        if len(self.memory) == 0:
            return np.array([])
        sample = random.sample(list(self.memory), min(n, len(self.memory)))
        with torch.no_grad():
            states_t = torch.tensor([s[0] for s in sample], dtype=torch.float32).to(self.device)
            return self.policy_net(states_t).cpu().numpy().flatten()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, buf):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "state_dim":  self.state_dim,
            "n_actions":  self.n_actions,
        }, buf)

    def load(self, buf):
        ckpt = torch.load(buf, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]

    # ── Private ───────────────────────────────────────────────────────────────

    def _learn(self):
        batch = random.sample(self.memory, self.batch_size)
        st, ac, rw, nst, dn = zip(*batch)

        st_t  = torch.tensor(st, dtype=torch.float32).to(self.device)
        ac_t  = torch.tensor(ac, dtype=torch.long).unsqueeze(1).to(self.device)
        rw_t  = torch.tensor(rw, dtype=torch.float32).to(self.device)
        nst_t = torch.tensor(nst, dtype=torch.float32).to(self.device)
        dn_t  = torch.tensor(dn, dtype=torch.float32).to(self.device)

        # Current Q-values
        current_q = self.policy_net(st_t).gather(1, ac_t).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q   = self.target_net(nst_t).max(dim=1)[0]
            target_q = rw_t + self.gamma * next_q * (1.0 - dn_t)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._opt_steps += 1
        if self._opt_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

