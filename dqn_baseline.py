"""
dqn_baseline.py  —  Real DQN implementation for Scenario 4 baseline.

Fixes addressed:
  #4  — Replaces the static threshold lookup table with an actual
         Deep Q-Network (Q-network + replay buffer + ε-greedy + target net).

Requires:
    pip install torch torchvision   # Apple Silicon: use MPS backend automatically

The discretised action space maps an integer index → (mcs, pmi, power_db).
"""

from __future__ import annotations
import random
import math
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from config_loader import CFG

_DQN_CFG = CFG["scenario4"]["dqn"]

# ---------------------------------------------------------------------------
# Discrete action space
# ---------------------------------------------------------------------------
# We discretise to keep the Q-network output tractable.
# Full grid: MCS ∈ {0,4,8,12,16,20,24,28} × PMI ∈ {0,4,8,12} × power ∈ {10,15,20,25,30}
# = 8 × 4 × 5 = 160 actions

_MCS_VALS   = list(range(0, 29, 4))   # [0,4,8,12,16,20,24,28]
_PMI_VALS   = list(range(0, 16, 4))   # [0,4,8,12]
_POWER_VALS = [10.0, 15.0, 20.0, 25.0, 30.0]

_ACTIONS: list[dict] = [
    {"mcs": m, "pmi": p, "power_db": pw}
    for m  in _MCS_VALS
    for p  in _PMI_VALS
    for pw in _POWER_VALS
]
N_ACTIONS = len(_ACTIONS)   # 160

# Observation: [snr_db, bler, throughput_mbps, mobility_norm, channel_flag]
OBS_DIM = 5


def _encode_obs(obs) -> np.ndarray:
    """Encode a BaseEnv Observation into a fixed-size float32 vector."""
    channel_flag = 1.0 if getattr(obs, "channel_model", "TDL-C") == "TDL-D" else 0.0
    return np.array([
        obs.snr_db / 30.0,
        obs.bler,
        obs.throughput_mbps / 50.0,
        getattr(obs, "mobility_kmh", 0.0) / 120.0,
        channel_flag,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128),     nn.ReLU(),
                nn.Linear(128, n_actions),
            )

        def forward(self, x):
            return self.net(x)
else:
    QNetwork = None   # type: ignore


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self._buf.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Real DQN with:
      - Experience replay buffer
      - ε-greedy exploration with exponential decay
      - Soft target network updates every `target_update_freq` steps

    Usage:
        agent = DQNAgent()
        agent.train(env, episodes=20)       # pretrain before comparison
        action = agent.select_action(obs)   # deterministic (ε=0) inference
    """

    def __init__(self, seed: int = CFG["global"]["seed"]):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Run: pip install torch\n"
                "On Apple Silicon: pip install torch torchvision torchaudio"
            )

        # Device: use Apple MPS if available, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self._q      = QNetwork().to(self._device)
        self._q_tgt  = QNetwork().to(self._device)
        self._q_tgt.load_state_dict(self._q.state_dict())
        self._q_tgt.eval()

        self._opt    = optim.Adam(self._q.parameters(), lr=_DQN_CFG["learning_rate"])
        self._buf    = ReplayBuffer(_DQN_CFG["buffer_size"])
        self._gamma  = _DQN_CFG["gamma"]
        self._eps    = _DQN_CFG["epsilon_start"]
        self._eps_end   = _DQN_CFG["epsilon_end"]
        self._eps_decay = _DQN_CFG["epsilon_decay"]
        self._batch  = _DQN_CFG["batch_size"]
        self._tuf    = _DQN_CFG["target_update_freq"]
        self._steps  = 0
        self._losses: list[float] = []

    # -----------------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------------

    def select_action(self, obs, greedy: bool = False) -> dict:
        """Return the best action dict. If greedy=False, applies ε-greedy."""
        if not greedy and random.random() < self._eps:
            idx = random.randint(0, N_ACTIONS - 1)
        else:
            state_t = torch.tensor(_encode_obs(obs), device=self._device).unsqueeze(0)
            with torch.no_grad():
                idx = int(self._q(state_t).argmax(dim=1).item())
        return dict(_ACTIONS[idx]), idx

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, env, episodes: Optional[int] = None) -> list[float]:
        """
        Pretrain the DQN on the given environment.
        Returns per-episode total rewards.
        """
        n_ep = episodes or _DQN_CFG["pretrain_episodes"]
        ep_rewards = []

        for ep in range(n_ep):
            obs = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, idx = self.select_action(obs)
                next_obs, info = env.step(action)
                done = info.terminated or info.truncated
                self._buf.push(
                    _encode_obs(obs), idx, info.reward,
                    _encode_obs(next_obs), float(done)
                )
                obs = next_obs
                ep_reward += info.reward
                self._steps += 1

                # Update ε
                self._eps = self._eps_end + (
                    _DQN_CFG["epsilon_start"] - self._eps_end
                ) * math.exp(-self._steps / self._eps_decay)

                # Gradient step
                if len(self._buf) >= self._batch:
                    loss = self._update()
                    self._losses.append(loss)

                # Target network sync
                if self._steps % self._tuf == 0:
                    self._q_tgt.load_state_dict(self._q.state_dict())

            ep_rewards.append(ep_reward)

        return ep_rewards

    def _update(self) -> float:
        states, actions, rewards, next_states, dones = self._buf.sample(self._batch)
        s  = torch.tensor(states,      device=self._device)
        a  = torch.tensor(actions,     device=self._device)
        r  = torch.tensor(rewards,     device=self._device)
        s2 = torch.tensor(next_states, device=self._device)
        d  = torch.tensor(dones,       device=self._device)

        q_vals  = self._q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._q_tgt(s2).max(1)[0]
            target = r + self._gamma * q_next * (1.0 - d)

        loss = nn.functional.smooth_l1_loss(q_vals, target)
        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._q.parameters(), 10.0)
        self._opt.step()
        return float(loss.item())

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str = "logs/dqn_s4.pt") -> None:
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._q.state_dict(), path)

    def load(self, path: str = "logs/dqn_s4.pt") -> None:
        self._q.load_state_dict(torch.load(path, map_location=self._device))
        self._q_tgt.load_state_dict(self._q.state_dict())
