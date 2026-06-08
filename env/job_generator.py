"""
job_generator.py
----------------
Defines the Job data class and functions to generate a batch of jobs
for one factory episode.

Job Types (TIGHT DEADLINES — creates real scheduling pressure):
    A (Short):  processing_time 3-6 steps,  energy 0.1-0.3,  slack 1 step
    B (Medium): processing_time 7-12 steps, energy 0.3-0.6,  slack 2 steps
    C (Long):   processing_time 13-20 steps, energy 0.6-1.0, slack 3 steps
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
#  JOB DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Job:
    """Represents a single production job."""

    job_id: int
    job_type: str           # 'A', 'B', or 'C'
    arrival_time: int       # Step when the job becomes available
    processing_time: int    # Steps needed to finish on any machine
    deadline: int           # Must be completed by this step
    energy_cost: float      # Energy consumed per processing step (raw)
    priority: str = 'normal'   # 'normal' or 'high' (rush orders = 'high')

    # Set during simulation
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    assigned_machine: Optional[int] = None

    # ── derived properties ──────────────────

    @property
    def tardiness(self) -> int:
        """Steps late past deadline. 0 if on time or not yet complete."""
        if self.completion_time is None:
            return 0
        return max(0, self.completion_time - self.deadline)

    @property
    def is_complete(self) -> bool:
        return self.completion_time is not None

    @property
    def wait_time(self) -> int:
        """Steps between arrival and start of processing."""
        if self.start_time is None:
            return 0
        return max(0, self.start_time - self.arrival_time)

    def type_to_float(self) -> float:
        """Encode job type as a float for the observation vector."""
        return {'A': 0.0, 'B': 0.5, 'C': 1.0}.get(self.job_type, 0.0)

    def reset_assignment(self):
        """Clear assignment state (used when a machine breaks mid-job)."""
        self.start_time = None
        self.completion_time = None
        self.assigned_machine = None

    def __repr__(self):
        return (f"Job(id={self.job_id}, type={self.job_type}, "
                f"arr={self.arrival_time}, proc={self.processing_time}, "
                f"dl={self.deadline}, pri={self.priority})")


# ─────────────────────────────────────────────
#  JOB BATCH GENERATOR
# ─────────────────────────────────────────────

# Type configuration table
JOB_TYPE_CONFIG = {
    'A': {
        'proc_time_range': (3, 6),
        'energy_range':    (0.10, 0.30),
        'deadline_slack':  5,   # Picked first by EDD → wait ≈ 0-5 steps
    },
    'B': {
        'proc_time_range': (7, 12),
        'energy_range':    (0.30, 0.60),
        'deadline_slack':  8,   # Picked second by EDD → wait ≈ 5-8 steps
    },
    'C': {
        'proc_time_range': (13, 20),
        'energy_range':    (0.60, 1.00),
        'deadline_slack':  12,  # Picked last by EDD → wait ≈ 8-12 steps
    },
}

# Probability of each type being sampled
JOB_TYPE_PROBS = [0.40, 0.35, 0.25]   # A, B, C


def create_job_batch(
    n_jobs: int,
    seed: int = None,
    config: dict = None,
) -> List[Job]:
    """
    Generate a batch of n_jobs Job objects for one episode.

    Args:
        n_jobs:  Number of jobs to create (typically 6).
        seed:    Random seed for reproducibility. None = random.
        config:  Dictionary with 'max_steps' key (default 100).

    Returns:
        List of Job objects sorted by arrival_time ascending.
    """
    rng = np.random.RandomState(seed)

    if config is None:
        config = {}
    max_steps = config.get('max_steps', 100)

    # Jobs arrive in the first 5% of the episode — burst creates real queue pressure
    # 6 jobs in 5 steps → 3 machines grab 3, other 3 queue up → FCFS vs EDD diverge
    arrival_window = max(1, int(max_steps * 0.05))

    jobs: List[Job] = []

    for i in range(n_jobs):
        # ── Sample type ───────────────────────────────────────────
        job_type = rng.choice(['A', 'B', 'C'], p=JOB_TYPE_PROBS)
        cfg = JOB_TYPE_CONFIG[job_type]

        # ── Processing time ───────────────────────────────────────
        lo, hi = cfg['proc_time_range']
        processing_time = int(rng.randint(lo, hi + 1))

        # ── Arrival time ──────────────────────────────────────────
        # Spread arrivals evenly across the window with some noise
        base_arrival = int((i / n_jobs) * arrival_window)
        jitter = int(rng.randint(0, max(1, arrival_window // n_jobs)))
        arrival_time = min(base_arrival + jitter, arrival_window)

        # ── Deadline ──────────────────────────────────────────────
        slack = cfg['deadline_slack']
        deadline = arrival_time + processing_time + slack

        # ── Energy cost ───────────────────────────────────────────
        e_lo, e_hi = cfg['energy_range']
        energy_cost = round(float(rng.uniform(e_lo, e_hi)), 3)

        jobs.append(Job(
            job_id=i,
            job_type=job_type,
            arrival_time=arrival_time,
            processing_time=processing_time,
            deadline=deadline,
            energy_cost=energy_cost,
        ))

    # Sort by arrival time so slot 0 = earliest arriving
    jobs.sort(key=lambda j: j.arrival_time)

    # ── Validation ────────────────────────────────────────────────
    for job in jobs:
        assert job.deadline > job.arrival_time + job.processing_time, (
            f"INVALID JOB: {job} — deadline must be > arrival + proc_time"
        )
        assert 0 < job.processing_time <= 20, (
            f"INVALID processing_time for {job}"
        )
        assert 0.0 < job.energy_cost <= 1.0, (
            f"INVALID energy_cost for {job}"
        )

    return jobs


# ─────────────────────────────────────────────
#  RUSH ORDER FACTORY
# ─────────────────────────────────────────────

def create_rush_order(current_step: int, job_id: int) -> Job:
    """
    Create a high-priority rush order that arrives immediately.
    Rush orders always have a tight deadline (slack 0-2 steps).

    Args:
        current_step: Current simulation time step.
        job_id:       Unique ID to assign (use a counter >= 1000).

    Returns:
        A Job with priority='high'.
    """
    rng = np.random.RandomState()   # Fresh random state for runtime events

    job_type = rng.choice(['A', 'B', 'C'], p=JOB_TYPE_PROBS)
    cfg = JOB_TYPE_CONFIG[job_type]

    lo, hi = cfg['proc_time_range']
    processing_time = int(rng.randint(lo, hi + 1))

    e_lo, e_hi = cfg['energy_range']
    energy_cost = round(float(rng.uniform(e_lo, e_hi)), 3)

    # Tight deadline — only 0 to 2 extra steps of slack
    slack = int(rng.randint(0, 3))
    deadline = current_step + processing_time + slack

    return Job(
        job_id=job_id,
        job_type=job_type,
        arrival_time=current_step,
        processing_time=processing_time,
        deadline=deadline,
        energy_cost=energy_cost,
        priority='high',
    )