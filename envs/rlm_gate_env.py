from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RLMRewardConfig:
    """Reward terms for event-level gating."""

    skip_penalty_bps: float = 0.0
    take_trade_penalty_bps: float = 0.0
    reward_clip_bps: float = 500.0


class RLMGateEnv(gym.Env[np.ndarray, int]):
    """
    Contextual event-gating environment.

    - Observation: event feature vector (market features by default).
    - Actions: 0=skip, 1=take.
    - Reward: realized event pnl in bps (for action=take) with optional penalties.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: np.ndarray,
        realized_pnl_bps: np.ndarray,
        reward_config: Optional[RLMRewardConfig] = None,
    ) -> None:
        super().__init__()
        obs = np.asarray(observations, dtype=np.float32)
        pnl = np.asarray(realized_pnl_bps, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"observations must be 2D [n_events, n_features], got shape={obs.shape}")
        if pnl.ndim != 1:
            raise ValueError(f"realized_pnl_bps must be 1D [n_events], got shape={pnl.shape}")
        if obs.shape[0] != pnl.shape[0]:
            raise ValueError("observations and realized_pnl_bps must have matching event counts")
        if obs.shape[0] == 0:
            raise ValueError("RLMGateEnv requires at least one event")

        self._obs = obs
        self._pnl_bps = pnl
        self._n_events = int(obs.shape[0])
        self._n_features = int(obs.shape[1])
        self._cfg = reward_config or RLMRewardConfig()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._n_features,),
            dtype=np.float32,
        )

        self._idx = 0
        self._taken = 0
        self._reward_sum = 0.0

    def _obs_at(self, idx: int) -> np.ndarray:
        if idx < self._n_events:
            return self._obs[idx]
        return np.zeros((self._n_features,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._idx = 0
        self._taken = 0
        self._reward_sum = 0.0
        return self._obs_at(0), {"n_events": self._n_events}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._idx >= self._n_events:
            return self._obs_at(self._idx), 0.0, True, False, {"n_events": self._n_events}

        a = int(action)
        pnl_bps = float(self._pnl_bps[self._idx])
        if a == 1:
            reward = pnl_bps - float(self._cfg.take_trade_penalty_bps)
            self._taken += 1
        else:
            reward = -float(self._cfg.skip_penalty_bps)
        clip = float(self._cfg.reward_clip_bps)
        if clip > 0:
            reward = float(np.clip(reward, -clip, clip))

        self._reward_sum += reward
        self._idx += 1
        terminated = self._idx >= self._n_events
        truncated = False
        info: Dict[str, Any] = {
            "event_index": int(self._idx - 1),
            "action": a,
            "event_pnl_bps": pnl_bps,
            "taken_count": int(self._taken),
            "reward_sum": float(self._reward_sum),
            "n_events": int(self._n_events),
        }
        return self._obs_at(self._idx), float(reward), bool(terminated), bool(truncated), info

