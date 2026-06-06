"""
networks.py
-----------
Neural network architectures for MAPPO.

ActorNetwork  (one per machine):
    Input:  38-dim local observation
    Output: 7-dim action probability distribution (Softmax)
    Params: ~50,000 per agent

CriticNetwork (one shared):
    Input:  114-dim global state (3 agents × 38 dims concatenated)
    Output: scalar value estimate V(s)
    Params: ~200,000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """
    Policy network for one machine agent.

    Converts a local 38-dim observation into action probabilities.
    Invalid actions are masked to -inf before softmax so the agent
    never selects an empty job slot or a non-idle machine action.
    """

    def __init__(self, obs_dim: int = 38, n_actions: int = 7, hidden: int = 128):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.hidden    = hidden

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # Initialise weights for stable early training
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Smaller init for output layer — action distribution starts near uniform
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            obs:         Tensor shape (batch, obs_dim)  — normalized floats
            action_mask: Tensor shape (batch, n_actions) — True = valid action

        Returns:
            probs: Tensor shape (batch, n_actions) — probability distribution
        """
        logits = self.net(obs)

        if action_mask is not None:
            # Replace invalid action logits with -inf so softmax gives 0 prob
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return F.softmax(logits, dim=-1)

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Sample an action and return its log probability.
        Used during rollout collection.
        """
        probs    = self.forward(obs, action_mask)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Evaluate log probs and entropy for given obs-action pairs.
        Used during PPO update.
        """
        probs    = self.forward(obs, action_mask)
        dist     = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters())
        return (
            f"ActorNetwork(obs={self.obs_dim} → hidden={self.hidden}×2 "
            f"→ actions={self.n_actions}) [{params:,} params]"
        )


class CriticNetwork(nn.Module):
    """
    Centralized value function (shared across all agents).

    Input is the GLOBAL state — all agents' observations concatenated.
    Only used during TRAINING (CTDE paradigm).
    At inference time, actors run independently without the critic.
    """

    def __init__(self, global_obs_dim: int = 114, hidden: int = 256):
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.hidden         = hidden

        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),   # scalar V(s)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: Tensor shape (batch, global_obs_dim)

        Returns:
            value: Tensor shape (batch, 1)
        """
        return self.net(global_state)

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters())
        return (
            f"CriticNetwork(global_obs={self.global_obs_dim} → "
            f"hidden={self.hidden}×2 → 1) [{params:,} params]"
        )