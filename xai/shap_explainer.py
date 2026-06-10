"""
shap_explainer.py
-----------------
SHAP-based explainability for MAPPO scheduling agents.

Uses GradientExplainer with logit (pre-softmax) output.
This avoids the softmax cancellation that makes DeepExplainer
return near-zero SHAP values on probability outputs.

Root cause of previous 0.0% issue:
    - DeepExplainer on softmax output: background states also
      choose action 0 at high probability (sorted queue → slot 0
      is usually best even for random policy), so SHAP sees
      zero marginal contribution for every feature.
    - Fix: explain logits (unbounded), not probabilities (0-1).

Feature layout (38-dimensional observation vector):
    [0-8]   Machine Status     — One-hot: Idle/Busy/Broken for M1, M2, M3
    [9-11]  Machine Load       — Normalized remaining processing steps
    [12-15] Job Slot 0 (Most Urgent) — Type, Proc, Deadline, Energy
    [16-19] Job Slot 1
    [20-23] Job Slot 2
    [24-27] Job Slot 3
    [28-31] Job Slot 4
    [32-35] Job Slot 5 (Least Urgent)
    [36]    Global Clock
    [37]    Energy Price
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import shap
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.networks import ActorNetwork


# ─────────────────────────────────────────────────────────────────────
#  FEATURE METADATA
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Machine Status  [0-8]
    'M1: Idle',  'M1: Busy',  'M1: Broken',
    'M2: Idle',  'M2: Busy',  'M2: Broken',
    'M3: Idle',  'M3: Busy',  'M3: Broken',
    # Machine Load  [9-11]
    'M1: Remaining Steps',
    'M2: Remaining Steps',
    'M3: Remaining Steps',
    # Job Slot 0 — Most Urgent  [12-15]
    'Job0: Type',  'Job0: Proc Time',  'Job0: Deadline Urgency',  'Job0: Energy Cost',
    # Job Slot 1  [16-19]
    'Job1: Type',  'Job1: Proc Time',  'Job1: Deadline Urgency',  'Job1: Energy Cost',
    # Job Slot 2  [20-23]
    'Job2: Type',  'Job2: Proc Time',  'Job2: Deadline Urgency',  'Job2: Energy Cost',
    # Job Slot 3  [24-27]
    'Job3: Type',  'Job3: Proc Time',  'Job3: Deadline Urgency',  'Job3: Energy Cost',
    # Job Slot 4  [28-31]
    'Job4: Type',  'Job4: Proc Time',  'Job4: Deadline Urgency',  'Job4: Energy Cost',
    # Job Slot 5 — Least Urgent  [32-35]
    'Job5: Type',  'Job5: Proc Time',  'Job5: Deadline Urgency',  'Job5: Energy Cost',
    # Context  [36-37]
    'Global Clock',
    'Energy Price',
]

FEATURE_GROUPS = {
    'Machine Status':       list(range(0, 9)),
    'Machine Load':         list(range(9, 12)),
    'Job 0 (Most Urgent)':  list(range(12, 16)),
    'Job 1':                list(range(16, 20)),
    'Jobs 2-5':             list(range(20, 36)),
    'Time Context':         [36],
    'Energy Price':         [37],
}

ACTION_LABELS = {
    0: 'Assign Job Slot 0 (Most Urgent)',
    1: 'Assign Job Slot 1',
    2: 'Assign Job Slot 2',
    3: 'Assign Job Slot 3',
    4: 'Assign Job Slot 4',
    5: 'Assign Job Slot 5 (Least Urgent)',
    6: 'WAIT (do not assign any job)',
}


# ─────────────────────────────────────────────────────────────────────
#  PYTORCH MODEL WRAPPER — returns LOGITS, not probabilities
# ─────────────────────────────────────────────────────────────────────

class _LogitActor(nn.Module):
    """
    Wraps ActorNetwork to return raw logits (before softmax).

    WHY THIS MATTERS:
        SHAP on softmax probabilities suffers from cancellation:
        gradients of p[i] w.r.t. inputs include negative contributions
        from all other actions (because softmax is normalized), which
        cancel out the positive signal we want.

        Logits are unbounded and independent across actions — SHAP
        on logit[i] measures exactly "how much does each input push
        the model toward action i?" without cross-action cancellation.
    """
    def __init__(self, actor: ActorNetwork):
        super().__init__()
        self.net = actor.net   # the Sequential inside ActorNetwork

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # shape (batch, 7) — raw logits


# ─────────────────────────────────────────────────────────────────────
#  SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Generates human-readable explanations for MAPPO scheduling decisions.

    Uses GradientExplainer on logit output (fast, reliable, no cancellation).

    Example:
        explainer = SHAPExplainer(agent.actors[0], env_config)
        obs       = env.reset(seed=0)[0]['machine_0']
        result    = explainer.explain(obs)
        print(result['narrative'])
    """

    def __init__(
        self,
        actor_network: ActorNetwork,
        env_config:    dict = None,
        n_background:  int  = 100,
        random_seed:   int  = 42,
        verbose:       bool = True,
    ):
        self.actor      = actor_network
        self.actor.eval()
        self.n_bg       = n_background
        self.seed       = random_seed
        self.verbose    = verbose

        if env_config is None:
            import json
            with open(os.path.join(PROJECT_ROOT, 'config.json')) as f:
                env_config = json.load(f)
        self.env_config = env_config

        if verbose:
            print(f"Initializing SHAPExplainer (GradientExplainer on logits)...")
            print(f"  Collecting {n_background} background states...",
                  end=' ', flush=True)

        self.background_np = self._build_background(n_background)

        if verbose:
            print(f"done  shape={self.background_np.shape}")

        self._init_explainer()

        if verbose:
            print(f"  Explainer : {self.explainer_type}")
            print("SHAPExplainer ready.\n")

    # ──────────────────────────────────────────────────────────────
    #  BACKGROUND COLLECTION
    # ──────────────────────────────────────────────────────────────

    def _build_background(self, n_samples: int) -> np.ndarray:
        """
        Collect a diverse set of factory states as SHAP background.

        KEY: We collect observations from ACROSS the episode
        (not just at step 0) to capture a wide range of feature values.
        We also collect from episodes with both busy and idle machines
        to maximize diversity and avoid the background being dominated
        by a single "typical" state.
        """
        from env.factory_gym import FactoryGym

        env      = FactoryGym(self.env_config)
        rng      = np.random.RandomState(self.seed)
        obs_list = []
        episode  = 0

        while len(obs_list) < n_samples:
            obs, _ = env.reset(seed=episode * 7 + 3)
            episode += 1
            step    = 0

            while len(obs_list) < n_samples:
                # Collect from EVERY step (not just first) for diversity
                obs_list.append(obs['machine_0'].copy())
                if len(obs_list) >= n_samples:
                    break

                # Random valid action
                actions = {}
                for i, aid in enumerate(env.agents):
                    mask  = env.get_action_mask(i)
                    valid = np.where(mask)[0]
                    actions[aid] = int(rng.choice(valid))

                obs, _, terms, truncs, _ = env.step(actions)
                step += 1
                if terms['__all__'] or truncs['__all__']:
                    break

        bg = np.array(obs_list[:n_samples], dtype=np.float32)
        return bg

    # ──────────────────────────────────────────────────────────────
    #  EXPLAINER INIT
    # ──────────────────────────────────────────────────────────────

    def _init_explainer(self):
        """
        Initialize GradientExplainer on the logit output.
        Falls back to integrated gradients if SHAP is unavailable.
        """
        self._logit_model = _LogitActor(self.actor)
        self._logit_model.eval()
        bg_tensor = torch.FloatTensor(self.background_np)

        try:
            self._ge = shap.GradientExplainer(
                self._logit_model, bg_tensor
            )
            self.explainer_type = 'GradientExplainer (logits)'
        except Exception as e:
            if self.verbose:
                print(f"  GradientExplainer unavailable ({e.__class__.__name__}), "
                      f"using IntegratedGradients fallback")
            self._ge = None
            self.explainer_type = 'IntegratedGradients (fallback)'

    # ──────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def explain(
        self,
        observation: np.ndarray,
        n_samples:   int = 50,
    ) -> dict:
        """
        Compute SHAP explanation for one agent observation.

        Args:
            observation : np.ndarray shape (38,)
            n_samples   : Samples for IntegratedGradients fallback (ignored for Gradient)

        Returns dict:
            chosen_action, action_label, action_prob, action_probs,
            shap_values (38,), top_features, groups, narrative
        """
        assert observation.shape == (38,), \
            f"Expected (38,) observation, got {observation.shape}"

        obs_2d = observation.reshape(1, -1).astype(np.float32)

        # Get action probabilities from actor (with softmax)
        with torch.no_grad():
            probs_t = self.actor(
                torch.FloatTensor(obs_2d), action_mask=None
            )
        probs_np      = probs_t.numpy()[0]
        chosen_action = int(np.argmax(probs_np))
        action_prob   = float(probs_np[chosen_action])

        # Compute SHAP values for chosen action's LOGIT
        raw_shap = self._compute_shap(obs_2d, chosen_action, n_samples)

        top_features = self._get_top_features(raw_shap, n=6)
        groups       = self._aggregate_groups(raw_shap)
        narrative    = self._build_narrative(
            chosen_action, action_prob, top_features, groups
        )

        return {
            'chosen_action': chosen_action,
            'action_label':  ACTION_LABELS.get(chosen_action,
                             f'Action {chosen_action}'),
            'action_prob':   action_prob,
            'action_probs':  probs_np.tolist(),
            'shap_values':   raw_shap,
            'top_features':  top_features,
            'groups':        groups,
            'narrative':     narrative,
        }

    # ──────────────────────────────────────────────────────────────
    #  SHAP COMPUTATION
    # ──────────────────────────────────────────────────────────────

    def _compute_shap(
        self,
        obs_2d:        np.ndarray,
        chosen_action: int,
        n_samples:     int,
    ) -> np.ndarray:
        """Route to GradientExplainer or IntegratedGradients fallback."""
        if self._ge is not None:
            return self._gradient_shap(obs_2d, chosen_action)
        else:
            return self._integrated_gradients(obs_2d, chosen_action, n_samples)

    def _gradient_shap(
        self,
        obs_2d:        np.ndarray,
        chosen_action: int,
    ) -> np.ndarray:
        """
        GradientExplainer on logit output.
        Returns SHAP values for chosen action's logit.
        """
        obs_tensor = torch.FloatTensor(obs_2d)

        try:
            # shap_values returns list of n_actions arrays, each shape (1, 38)
            shap_vals = self._ge.shap_values(obs_tensor)

            if isinstance(shap_vals, list):
                arr = np.array(shap_vals[chosen_action]).flatten()
            else:
                # GradientExplainer for PyTorch models often returns (batch, features, actions)
                # Ensure we index correctly instead of flattening the whole multi-action array.
                sv = np.array(shap_vals)
                if sv.ndim == 3 and sv.shape[-1] > 1: # e.g. (1, 38, 7)
                    arr = sv[0, :, chosen_action].flatten()
                elif sv.ndim == 3 and sv.shape[1] > 1: # e.g. (7, 1, 38)
                    arr = sv[chosen_action, 0, :].flatten()
                else:
                    arr = sv.flatten()

            arr = arr[:38] if len(arr) >= 38 else np.pad(arr, (0, 38 - len(arr)))

            # If all zeros, fall back to integrated gradients
            if np.abs(arr).sum() < 1e-10:
                if self.verbose:
                    print("  GradientExplainer returned zeros — "
                          "falling back to IntegratedGradients")
                return self._integrated_gradients(obs_2d, chosen_action, 50)

            return arr

        except Exception as e:
            return self._integrated_gradients(obs_2d, chosen_action, 50)

    def _integrated_gradients(
        self,
        obs_2d:        np.ndarray,
        chosen_action: int,
        n_steps:       int = 50,
    ) -> np.ndarray:
        """
        Integrated Gradients attribution (Sundararajan et al., 2017).

        Axioms: completeness (attributions sum to output difference)
                sensitivity, implementation invariance.

        Baseline = mean of background (not zeros) to stay in-distribution.
        """
        obs_tensor      = torch.FloatTensor(obs_2d)
        baseline_tensor = torch.FloatTensor(
            self.background_np.mean(axis=0, keepdims=True)
        )

        # Linearly interpolate from baseline to observation
        alphas       = torch.linspace(0.0, 1.0, n_steps)
        interpolated = torch.cat([
            baseline_tensor + alpha * (obs_tensor - baseline_tensor)
            for alpha in alphas
        ], dim=0)   # shape (n_steps, 38)

        interpolated.requires_grad_(True)

        # Forward pass through logit model
        logits        = self._logit_model(interpolated)  # (n_steps, 7)
        chosen_logits = logits[:, chosen_action].sum()
        chosen_logits.backward()

        grads = interpolated.grad.detach().numpy()  # (n_steps, 38)

        # IG: average gradient × (input − baseline)
        avg_grads   = grads.mean(axis=0)   # (38,)
        delta       = (obs_2d[0]
                       - baseline_tensor.detach().numpy()[0])  # (38,)
        attribution = avg_grads * delta    # (38,)

        return attribution.astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    #  FORMATTING HELPERS
    # ──────────────────────────────────────────────────────────────

    def _get_top_features(
        self,
        shap_values: np.ndarray,
        n:           int = 6,
    ) -> List[Tuple[str, float]]:
        total = float(np.abs(shap_values).sum())
        if total < 1e-12:
            return [(FEATURE_NAMES[i], 0.0) for i in range(min(n, 38))]

        abs_sv  = np.abs(shap_values)
        top_idx = np.argsort(abs_sv)[::-1][:n]
        return [
            (
                FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES)
                else f'Feature_{idx}',
                round(float(abs_sv[idx] / total * 100), 1),
            )
            for idx in top_idx
        ]

    def _aggregate_groups(
        self,
        shap_values: np.ndarray,
    ) -> Dict[str, float]:
        total = float(np.abs(shap_values).sum())
        if total < 1e-12:
            return {name: 0.0 for name in FEATURE_GROUPS}

        return {
            name: round(
                float(sum(abs(shap_values[i])
                          for i in indices
                          if i < len(shap_values))) / total * 100,
                1,
            )
            for name, indices in FEATURE_GROUPS.items()
        }

    def _build_narrative(
        self,
        chosen_action: int,
        action_prob:   float,
        top_features:  List[Tuple[str, float]],
        groups:        Dict[str, float],
    ) -> str:
        label = ACTION_LABELS.get(chosen_action, f'Action {chosen_action}')
        lines = [
            f"Decision: {label}",
            f"Confidence: {action_prob*100:.1f}%",
            f"",
            f"Top reasons (by feature importance):",
        ]
        for rank, (fname, pct) in enumerate(top_features[:4], 1):
            lines.append(f"  {rank}. {fname:<38} {pct:>5.1f}%")

        lines += ["", "Importance by group:"]
        sorted_g = sorted(groups.items(), key=lambda x: x[1], reverse=True)
        for gname, pct in sorted_g[:4]:
            if pct > 0.5:
                lines.append(f"  {gname:<28} {pct:>5.1f}%")

        return '\n'.join(lines)

    # ──────────────────────────────────────────────────────────────
    #  CONVENIENCE
    # ──────────────────────────────────────────────────────────────

    def explain_action(
        self,
        obs_dict:   dict,
        agent_idx:  int = 0,
    ) -> dict:
        """Explain from full obs dict (output of env.step/reset)."""
        return self.explain(obs_dict[f'machine_{agent_idx}'])

    def get_top_group(self, explanation: dict) -> str:
        return max(explanation['groups'], key=explanation['groups'].get)

    def __repr__(self) -> str:
        return (f"SHAPExplainer(background={self.n_bg}, "
                f"type={self.explainer_type})")