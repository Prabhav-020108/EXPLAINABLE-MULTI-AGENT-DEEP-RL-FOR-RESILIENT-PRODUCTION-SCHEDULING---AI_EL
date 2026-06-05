"""
reward.py
---------
Reward calculation for the FactoryGym environment.

Total reward formula:
    r = 0.30 * r_complete  +  0.40 * r_tardy  +  0.20 * r_energy  +  0.10 * r_idle

Weights are configurable through config.json.
"""

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from env.job_generator import Job


class RewardCalculator:
    """
    Computes the scalar step reward from environment events.

    Weights default to the blueprint specification:
        completion  = 0.30
        tardiness   = 0.40
        energy      = 0.20
        idle        = 0.10
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        w = config.get('reward_weights', {})
        self.w_completion = w.get('completion', 0.30)
        self.w_tardiness  = w.get('tardiness',  0.40)
        self.w_energy     = w.get('energy',     0.20)
        self.w_idle       = w.get('idle',       0.10)

        # Maximum possible energy per step (used for normalization)
        # 3 machines × max_energy_cost (1.0) × spike_factor (3.0) = 9.0
        self.max_energy_per_step = config.get('max_energy_per_step', 9.0)

        # Validate weights sum to 1.0
        total = self.w_completion + self.w_tardiness + self.w_energy + self.w_idle
        assert abs(total - 1.0) < 1e-6, (
            f"Reward weights must sum to 1.0, got {total:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────
    #  MAIN COMPUTE METHOD
    # ──────────────────────────────────────────────────────────────────

    def compute(
        self,
        jobs_completed_this_step: List,
        tardiness_this_step: float,
        energy_used_this_step: float,
        n_idle_machines: int,
        max_deadline: int = 100,
        disruption_recovery: bool = False,
    ) -> float:
        """
        Compute total scalar reward for one time step.

        Args:
            jobs_completed_this_step: List of Job objects finished this step.
            tardiness_this_step:      Total tardiness steps accumulated this step.
            energy_used_this_step:    Total energy consumed this step (summed across machines).
            n_idle_machines:          Number of machines doing nothing this step.
            max_deadline:             Used for normalizing tardiness (= max_steps).
            disruption_recovery:      True if a recovery happened after a breakdown.

        Returns:
            float: scalar reward (can be negative or positive)
        """
        r_complete = self._completion_reward(jobs_completed_this_step, disruption_recovery)
        r_tardy    = self._tardiness_penalty(tardiness_this_step, max_deadline)
        r_energy   = self._energy_penalty(energy_used_this_step)
        r_idle     = self._idle_penalty(n_idle_machines)

        r_total = (
            self.w_completion * r_complete
            + self.w_tardiness  * r_tardy
            + self.w_energy     * r_energy
            + self.w_idle       * r_idle
        )

        # Safety guards
        if np.isnan(r_total):
            raise ValueError(
                f"Reward is NaN! Components: complete={r_complete}, "
                f"tardy={r_tardy}, energy={r_energy}, idle={r_idle}"
            )
        if np.isinf(r_total):
            raise ValueError(f"Reward is Infinite! r_total={r_total}")

        return float(r_total)

    # ──────────────────────────────────────────────────────────────────
    #  COMPONENT METHODS  (each returns a pre-weighted component value)
    # ──────────────────────────────────────────────────────────────────

    def _completion_reward(self, completed_jobs: List, recovery: bool) -> float:
        """
        +5.0 for each job completed on time.
        +2.0 extra bonus if a job was recovered after a breakdown.
        """
        r = 0.0
        for job in completed_jobs:
            if job.tardiness == 0:
                r += 5.0    # on-time bonus
        if recovery and completed_jobs:
            r += 2.0        # disruption recovery bonus
        return r

    def _tardiness_penalty(self, tardiness_steps: float, max_deadline: int) -> float:
        """
        Normalized tardiness penalty.
        -10 * (total_late_steps / max_deadline)
        Clipped to [-10, 0].
        """
        if max_deadline <= 0 or tardiness_steps <= 0:
            return 0.0
        penalty = -10.0 * (tardiness_steps / max_deadline)
        return float(np.clip(penalty, -10.0, 0.0))

    def _energy_penalty(self, energy_used: float) -> float:
        """
        Normalized energy penalty.
        -(energy_used / max_energy_per_step)
        Clipped to [-1, 0].
        """
        if energy_used <= 0:
            return 0.0
        penalty = -(energy_used / self.max_energy_per_step)
        return float(np.clip(penalty, -1.0, 0.0))

    def _idle_penalty(self, n_idle: int) -> float:
        """
        -0.1 per idle machine per step.
        """
        return -0.1 * max(0, n_idle)

    def __repr__(self):
        return (
            f"RewardCalculator(w_comp={self.w_completion}, w_tard={self.w_tardiness}, "
            f"w_energy={self.w_energy}, w_idle={self.w_idle})"
        )