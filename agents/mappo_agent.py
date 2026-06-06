"""
mappo_agent.py
--------------
Multi-Agent PPO (MAPPO) implementation.

Key components:
    RolloutBuffer  — Stores one rollout of experience for all agents
    MAPPOAgent     — Wraps actors + critic, handles predict() and update()

Training paradigm:
    CTDE — Centralized Training (critic sees global state),
           Decentralized Execution (actors use only local obs)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from agents.networks import ActorNetwork, CriticNetwork


# ════════════════════════════════════════════════════════════════════
#  ROLLOUT BUFFER
# ════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stores experience tuples collected during one rollout.

    One buffer holds data for ALL agents simultaneously.
    Each call to .add() stores one time step across all agents.
    """

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int):
        self.n_agents  = n_agents
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.clear()

    def clear(self):
        """Reset buffer for a new rollout."""
        # Per-agent lists (index i = machine i)
        self.obs       = [[] for _ in range(self.n_agents)]
        self.actions   = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.rewards   = [[] for _ in range(self.n_agents)]
        self.masks     = [[] for _ in range(self.n_agents)]

        # Shared across agents
        self.dones         = []
        self.global_states = []   # concatenated obs of all agents
        self.values        = []   # critic V(s) at each step

    def add(
        self,
        obs_dict:      Dict[str, np.ndarray],
        actions_dict:  Dict[str, int],
        log_probs_dict:Dict[str, float],
        rewards_dict:  Dict[str, float],
        done:          bool,
        global_state:  np.ndarray,
        value:         float,
        masks_dict:    Dict[str, np.ndarray] = None,
    ):
        """Store one transition for all agents."""
        for i in range(self.n_agents):
            aid = f'machine_{i}'
            self.obs[i].append(obs_dict[aid].astype(np.float32))
            self.actions[i].append(int(actions_dict[aid]))
            self.log_probs[i].append(float(log_probs_dict[aid]))
            self.rewards[i].append(float(rewards_dict[aid]))

            if masks_dict and aid in masks_dict:
                self.masks[i].append(masks_dict[aid].astype(bool))
            else:
                # Default: all actions valid
                self.masks[i].append(np.ones(self.n_actions, dtype=bool))

        self.dones.append(float(done))
        self.global_states.append(global_state.astype(np.float32))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.values)

    def get_agent_tensors(
        self, agent_idx: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Return (obs, actions, log_probs, masks) tensors for one agent."""
        obs       = torch.FloatTensor(np.array(self.obs[agent_idx])).to(device)
        actions   = torch.LongTensor(np.array(self.actions[agent_idx])).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs[agent_idx])).to(device)
        masks     = torch.BoolTensor(np.array(self.masks[agent_idx])).to(device)
        return obs, actions, log_probs, masks

    def get_shared_tensors(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (global_states, values, dones) tensors."""
        states = torch.FloatTensor(np.array(self.global_states)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        dones  = torch.FloatTensor(np.array(self.dones)).to(device)
        return states, values, dones


# ════════════════════════════════════════════════════════════════════
#  MAPPO AGENT
# ════════════════════════════════════════════════════════════════════

class MAPPOAgent:
    """
    Multi-Agent PPO agent for factory scheduling.

    Public interface:
        predict(obs_dict, masks_dict)          → actions, log_probs, probs
        predict_with_value(obs_dict, masks_dict) → actions, log_probs, global_state, value
        update(next_value)                     → loss metrics dict
        save(path)
        load(path)
        set_eval_mode() / set_train_mode()
    """

    # Observation dimensions
    OBS_DIM        = 38
    N_AGENTS       = 3
    GLOBAL_OBS_DIM = OBS_DIM * N_AGENTS   # 114

    def __init__(
        self,
        obs_dim:   int  = 38,
        n_actions: int  = 7,
        n_agents:  int  = 3,
        config:    dict = None,
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.n_agents  = n_agents

        if config is None:
            config = {}

        # ── Hyperparameters ───────────────────────────────────────
        self.lr            = config.get('learning_rate', 3e-4)
        self.gamma         = config.get('gamma',         0.99)
        self.lam           = config.get('lam',           0.95)
        self.clip_ratio    = config.get('clip_ratio',    0.2)
        self.entropy_coef  = config.get('entropy_coef',  0.01)
        self.value_coef    = config.get('value_coef',    0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.batch_size    = config.get('batch_size',    64)
        self.n_epochs      = config.get('n_epochs',      10)

        # ── Device ────────────────────────────────────────────────
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # ── Networks ──────────────────────────────────────────────
        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, n_actions).to(self.device)
            for _ in range(n_agents)
        ])
        self.critic = CriticNetwork(obs_dim * n_agents).to(self.device)

        # ── Optimizers ────────────────────────────────────────────
        all_actor_params = []
        for actor in self.actors:
            all_actor_params += list(actor.parameters())

        self.actor_optimizer  = optim.Adam(all_actor_params,            lr=self.lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),    lr=self.lr, eps=1e-5)

        # ── Rollout buffer ────────────────────────────────────────
        self.buffer = RolloutBuffer(n_agents, obs_dim, n_actions)

        # Print summary
        n_actor_params  = sum(p.numel() for p in self.actors[0].parameters())
        n_critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"\nMAPPOAgent — device: {self.device}")
        print(f"  {n_agents}× ActorNetwork  : {n_actor_params:,} params each")
        print(f"  1× CriticNetwork : {n_critic_params:,} params (shared)")
        print(f"  Hyperparams: lr={self.lr}, gamma={self.gamma}, "
              f"clip={self.clip_ratio}, epochs={self.n_epochs}\n")

    # ──────────────────────────────────────────────────────────────
    #  PREDICT (inference only)
    # ──────────────────────────────────────────────────────────────

    def predict(
        self,
        obs_dict:   Dict[str, np.ndarray],
        masks_dict: Dict[str, np.ndarray] = None,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Select actions for all agents (no gradient tracking).

        Args:
            obs_dict:   {agent_id: np.ndarray (38,)}
            masks_dict: {agent_id: np.ndarray (7,) bool}

        Returns:
            actions_dict:   {agent_id: int}
            log_probs_dict: {agent_id: float}
            probs_dict:     {agent_id: np.ndarray (7,)}
        """
        actions_dict   = {}
        log_probs_dict = {}
        probs_dict     = {}

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                aid = f'machine_{i}'

                obs_t = torch.FloatTensor(
                    obs_dict[aid]
                ).unsqueeze(0).to(self.device)

                mask_t = None
                if masks_dict and aid in masks_dict:
                    mask_t = torch.BoolTensor(
                        masks_dict[aid]
                    ).unsqueeze(0).to(self.device)

                action, log_prob, _, probs = actor.get_action_and_log_prob(obs_t, mask_t)

                actions_dict[aid]   = int(action.item())
                log_probs_dict[aid] = float(log_prob.item())
                probs_dict[aid]     = probs.squeeze(0).cpu().numpy()

        return actions_dict, log_probs_dict, probs_dict

    def predict_with_value(
        self,
        obs_dict:   Dict[str, np.ndarray],
        masks_dict: Dict[str, np.ndarray] = None,
    ) -> Tuple[Dict, Dict, np.ndarray, float]:
        """
        Like predict(), but also returns global_state and critic value.
        Used during rollout collection in training.
        """
        actions_dict   = {}
        log_probs_dict = {}

        global_state = np.concatenate([
            obs_dict[f'machine_{i}'] for i in range(self.n_agents)
        ]).astype(np.float32)

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                aid = f'machine_{i}'

                obs_t = torch.FloatTensor(
                    obs_dict[aid]
                ).unsqueeze(0).to(self.device)

                mask_t = None
                if masks_dict and aid in masks_dict:
                    mask_t = torch.BoolTensor(
                        masks_dict[aid]
                    ).unsqueeze(0).to(self.device)

                action, log_prob, _, _ = actor.get_action_and_log_prob(obs_t, mask_t)
                actions_dict[aid]   = int(action.item())
                log_probs_dict[aid] = float(log_prob.item())

            # Critic value for global state
            gs_t  = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = float(self.critic(gs_t).item())

        return actions_dict, log_probs_dict, global_state, value

    # ──────────────────────────────────────────────────────────────
    #  GENERALIZED ADVANTAGE ESTIMATION
    # ──────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        rewards:    List[float],
        values:     List[float],
        dones:      List[float],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        GAE formula:
            A_t = sum_{k=0}^{inf} (gamma * lambda)^k * delta_{t+k}
            delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)

        Args:
            rewards:    List of per-step rewards (length T)
            values:     List of critic value estimates (length T)
            dones:      List of done flags (length T)
            next_value: Critic value for state AFTER last step

        Returns:
            advantages: np.ndarray (T,) — normalized
            returns:    np.ndarray (T,) — advantages + values (for critic target)
        """
        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        # Extend values by one for bootstrapping
        values_ext = values + [next_value]

        for t in reversed(range(T)):
            delta = (
                rewards[t]
                + self.gamma * values_ext[t + 1] * (1.0 - dones[t])
                - values_ext[t]
            )
            gae          = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)

        # Normalize advantages for stable training
        adv_mean   = advantages.mean()
        adv_std    = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return advantages, returns

    # ──────────────────────────────────────────────────────────────
    #  PPO UPDATE
    # ──────────────────────────────────────────────────────────────

    def update(self, next_value: float) -> Dict[str, float]:
        """
        Run PPO update using the current rollout buffer.

        Steps:
            1. Compute GAE advantages from buffer rewards and values
            2. For n_epochs:
               a. Shuffle buffer indices
               b. For each mini-batch:
                  - Compute critic loss (MSE on returns)
                  - Compute actor loss (PPO clipped objective)
                  - Add entropy bonus for exploration
                  - Backprop + gradient clip + optimizer step
            3. Return mean losses for logging

        Args:
            next_value: Critic's V estimate for the state after the last buffer step.
        """
        # ── Compute advantages ────────────────────────────────────
        rewards = self.buffer.rewards[0]   # same reward for all agents (shared)
        dones   = self.buffer.dones
        values  = self.buffer.values

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t    = torch.FloatTensor(returns).to(self.device)

        global_states_t, _, _ = self.buffer.get_shared_tensors(self.device)

        T       = len(self.buffer)
        indices = np.arange(T)

        # Accumulate losses for logging
        actor_losses  = []
        critic_losses = []
        entropy_vals  = []

        # ── PPO epochs ────────────────────────────────────────────
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.batch_size):
                batch_idx = torch.LongTensor(
                    indices[start: start + self.batch_size]
                ).to(self.device)

                b_advantages = advantages_t[batch_idx]
                b_returns    = returns_t[batch_idx]
                b_states     = global_states_t[batch_idx]

                # ── Critic update ──────────────────────────────────
                v_pred      = self.critic(b_states).squeeze(-1)
                critic_loss = nn.MSELoss()(v_pred, b_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))

                # ── Actor update (all agents combined) ────────────
                total_actor_loss = torch.tensor(0.0).to(self.device)
                total_entropy    = torch.tensor(0.0).to(self.device)

                for i, actor in enumerate(self.actors):
                    obs_i, acts_i, old_lp_i, masks_i = self.buffer.get_agent_tensors(
                        i, self.device
                    )
                    obs_i    = obs_i[batch_idx]
                    acts_i   = acts_i[batch_idx]
                    old_lp_i = old_lp_i[batch_idx]
                    masks_i  = masks_i[batch_idx]

                    new_lp_i, ent_i = actor.evaluate_actions(obs_i, acts_i, masks_i)
                    entropy_mean    = ent_i.mean()

                    # PPO clipped objective
                    ratio = torch.exp(new_lp_i - old_lp_i)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * b_advantages
                    actor_loss_i = -torch.min(surr1, surr2).mean()

                    total_actor_loss = total_actor_loss + actor_loss_i
                    total_entropy    = total_entropy    + entropy_mean

                # Average across agents
                total_actor_loss = total_actor_loss / self.n_agents
                total_entropy    = total_entropy    / self.n_agents

                final_loss = total_actor_loss - self.entropy_coef * total_entropy

                self.actor_optimizer.zero_grad()
                final_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for actor in self.actors for p in actor.parameters()],
                    self.max_grad_norm,
                )
                self.actor_optimizer.step()

                actor_losses.append(float(total_actor_loss.item()))
                entropy_vals.append(float(total_entropy.item()))

        return {
            'actor_loss':  float(np.mean(actor_losses)),
            'critic_loss': float(np.mean(critic_losses)),
            'entropy':     float(np.mean(entropy_vals)),
        }

    # ──────────────────────────────────────────────────────────────
    #  SAVE / LOAD
    # ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save full model checkpoint to a .pth file."""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        checkpoint = {
            'actors':  [actor.state_dict() for actor in self.actors],
            'critic':  self.critic.state_dict(),
            'config': {
                'obs_dim':    self.obs_dim,
                'n_actions':  self.n_actions,
                'n_agents':   self.n_agents,
                'lr':         self.lr,
                'gamma':      self.gamma,
                'clip_ratio': self.clip_ratio,
                'entropy_coef': self.entropy_coef,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint from a .pth file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from: {path}")

    def set_eval_mode(self):
        """Switch to inference mode."""
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

    def set_train_mode(self):
        """Switch back to training mode."""
        for actor in self.actors:
            actor.train()
        self.critic.train()

    def __repr__(self):
        return (
            f"MAPPOAgent(agents={self.n_agents}, obs={self.obs_dim}, "
            f"actions={self.n_actions}, device={self.device})"
        )