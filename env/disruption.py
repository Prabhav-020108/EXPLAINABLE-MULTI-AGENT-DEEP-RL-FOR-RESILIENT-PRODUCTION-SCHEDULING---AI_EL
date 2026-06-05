"""
disruption.py
-------------
Manages random factory disruptions:
    1. Machine Breakdown  — machine stops for T_repair steps
    2. Rush Order         — urgent high-priority job inserted into queue
    3. Energy Price Spike — energy costs multiplied by spike_factor for N steps

Disruption rates are configurable through config.json.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ─────────────────────────────────────────────
#  DISRUPTION EVENT
# ─────────────────────────────────────────────

@dataclass
class DisruptionEvent:
    """Records one disruption event that occurred during simulation."""
    event_type: str          # 'breakdown' | 'rush_order' | 'energy_spike'
    step: int                # Step when it occurred
    details: Dict[str, Any]  # Type-specific info (machine_id, factor, etc.)

    def __repr__(self):
        return f"DisruptionEvent(type={self.event_type}, step={self.step}, {self.details})"


# ─────────────────────────────────────────────
#  DISRUPTION MANAGER
# ─────────────────────────────────────────────

class DisruptionManager:
    """
    Samples and tracks all disruption events during an episode.

    Usage in environment step():
        events = disruption_mgr.sample_disruptions(step, machines)
        for event in events:
            apply event to factory state
    """

    REPAIR_STEPS = 10   # How many steps a broken machine is unavailable

    def __init__(
        self,
        breakdown_rate: float = 0.02,
        rush_rate: float = 0.01,
        energy_spike_rate: float = 0.05,
        n_machines: int = 3,
        seed: int = None,
    ):
        """
        Args:
            breakdown_rate:     Probability of each non-broken machine breaking per step.
            rush_rate:          Probability of a rush order arriving per step.
            energy_spike_rate:  Probability of a price spike starting per step.
            n_machines:         Number of machines to monitor for breakdowns.
            seed:               Random seed for reproducibility.
        """
        self.breakdown_rate    = breakdown_rate
        self.rush_rate         = rush_rate
        self.energy_spike_rate = energy_spike_rate
        self.n_machines        = n_machines

        # Separate RNG so disruptions don't affect job generation seeds
        self.rng = np.random.RandomState(seed)

        # Energy spike state
        self.current_spike_factor: float = 1.0
        self.spike_remaining_steps: int  = 0

        # Event log for the current episode
        self.history: List[DisruptionEvent] = []

    # ──────────────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────────

    def sample_disruptions(
        self,
        step: int,
        machines: List[Dict],
    ) -> List[DisruptionEvent]:
        """
        Sample all disruption events for the current time step.

        Args:
            step:     Current simulation time step.
            machines: List of machine state dicts from FactoryGym.

        Returns:
            List of DisruptionEvent objects to apply (may be empty).
        """
        events: List[DisruptionEvent] = []

        # ── 1. Machine Breakdown ──────────────────────────────────
        for machine in machines:
            # Only idle or busy machines can break (not already broken)
            if machine['status'] in ('idle', 'busy'):
                if self.rng.random() < self.breakdown_rate:
                    evt = DisruptionEvent(
                        event_type='breakdown',
                        step=step,
                        details={
                            'machine_id':     machine['id'],
                            'repair_steps':   self.REPAIR_STEPS,
                            'was_processing': machine['current_job'] is not None,
                        }
                    )
                    events.append(evt)
                    self.history.append(evt)

        # ── 2. Rush Order ─────────────────────────────────────────
        if self.rng.random() < self.rush_rate:
            evt = DisruptionEvent(
                event_type='rush_order',
                step=step,
                details={}
            )
            events.append(evt)
            self.history.append(evt)

        # ── 3. Energy Price Spike ─────────────────────────────────
        if self.spike_remaining_steps > 0:
            # Active spike — count down
            self.spike_remaining_steps -= 1
            if self.spike_remaining_steps == 0:
                self.current_spike_factor = 1.0   # spike ends
        else:
            # No active spike — check if a new one starts
            if self.rng.random() < self.energy_spike_rate:
                factor   = round(float(self.rng.uniform(1.5, 3.0)), 2)
                duration = int(self.rng.randint(5, 16))   # 5-15 steps
                self.current_spike_factor    = factor
                self.spike_remaining_steps   = duration
                evt = DisruptionEvent(
                    event_type='energy_spike',
                    step=step,
                    details={'factor': factor, 'duration': duration}
                )
                events.append(evt)
                self.history.append(evt)

        return events

    def get_energy_multiplier(self) -> float:
        """Current energy price multiplier (1.0 = normal, up to 3.0 during spike)."""
        return self.current_spike_factor

    def is_spike_active(self) -> bool:
        return self.spike_remaining_steps > 0

    def reset(self):
        """Clear all state for a new episode."""
        self.history               = []
        self.current_spike_factor  = 1.0
        self.spike_remaining_steps = 0

    def get_breakdown_count(self) -> int:
        return sum(1 for e in self.history if e.event_type == 'breakdown')

    def get_rush_order_count(self) -> int:
        return sum(1 for e in self.history if e.event_type == 'rush_order')

    def __repr__(self):
        return (
            f"DisruptionManager(breakdown={self.breakdown_rate}, "
            f"rush={self.rush_rate}, spike={self.energy_spike_rate})"
        )