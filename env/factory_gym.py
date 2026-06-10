"""
factory_gym.py
--------------
FactoryGym: a multi-machine production scheduling environment
compatible with the Gymnasium (gym) API.

Three machines, six jobs per episode, with machine breakdowns,
rush orders, and energy price spikes.

Observation space:  Box(0.0, 1.0, shape=(38,), dtype=float32)
Action space:       Discrete(7) — select job slot 0-5, or 6=WAIT

The environment operates in MULTI-AGENT mode:
    step() accepts a dict  {agent_id: action_int}
    step() returns dicts   {agent_id: value}  for obs, reward, done, etc.
"""

import json
import os
import copy
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Any

from env.job_generator import Job, create_job_batch, create_rush_order
from env.reward import RewardCalculator
from env.disruption import DisruptionManager


class FactoryGym(gym.Env):
    """
    Multi-machine production scheduling environment.

    Three independent agents, one per machine.
    Each agent selects which available job to process, or waits.
    """

    metadata = {'render_modes': ['dict']}

    # ──────────────────────────────────────────────────────────────────
    #  INITIALISATION
    # ──────────────────────────────────────────────────────────────────

    def __init__(self, config: dict = None, render_mode: str = 'dict'):
        super().__init__()

        # ── Load configuration ────────────────────────────────────
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'config.json'
            )
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

        self.config = config
        self.n_machines  = config.get('n_machines',  3)
        self.n_jobs      = config.get('n_jobs',      6)
        self.max_steps   = config.get('max_steps',   100)
        self.render_mode = render_mode

        # ── Gymnasium spaces ──────────────────────────────────────
        # 38-dimensional normalized float observation
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(38,), dtype=np.float32
        )
        # n_jobs job slots (0..n_jobs-1) + 1 WAIT action
        self.action_space = gym.spaces.Discrete(self.n_jobs + 1)

        # ── Agent identifiers ─────────────────────────────────────
        self.agents      = [f'machine_{i}' for i in range(self.n_machines)]
        self.num_agents  = self.n_machines

        # ── Normalisation constants ───────────────────────────────
        self.MAX_PROC_TIME    = 20
        self.MAX_DEADLINE     = self.max_steps + 30
        self.MAX_ENERGY_COST  = 1.0
        self.MAX_SPIKE_FACTOR = 3.0

        # ── Sub-modules ───────────────────────────────────────────
        self.reward_calc = RewardCalculator(config)
        self.disruption_mgr = DisruptionManager(
            breakdown_rate    = config.get('breakdown_rate',     0.02),
            rush_rate         = config.get('rush_rate',          0.01),
            energy_spike_rate = config.get('energy_spike_rate',  0.05),
            n_machines        = self.n_machines,
        )

        # ── Episode state (initialised in reset) ──────────────────
        self.machines:        List[Dict] = []
        self.available_jobs:  List[Job]  = []   # arrived, not yet assigned
        self.pending_jobs:    List[Job]  = []   # not yet arrived
        self.all_jobs:        List[Job]  = []
        self.completed_jobs:  List[Job]  = []
        self.current_step:    int        = 0
        self.episode_id:      int        = 0
        self._rush_id_counter: int       = 1000
        self._episode_energy:  float     = 0.0

    # ──────────────────────────────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int = None,
        options: dict = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment for a new episode.

        Returns:
            observations: dict  {agent_id: np.ndarray shape (38,)}
            info:         dict  {episode metadata}
        """
        super().reset(seed=seed)
        self.current_step   = 0
        self.episode_id    += 1
        self._episode_energy = 0.0

        # ── Fresh machines ────────────────────────────────────────
        self.machines = [
            {
                'id':              i,
                'status':          'idle',     # 'idle' | 'busy' | 'broken'
                'current_job':     None,
                'remaining_steps': 0,
                'repair_countdown':0,
                'completed_jobs':  [],
            }
            for i in range(self.n_machines)
        ]

        # ── Generate jobs ─────────────────────────────────────────
        actual_seed = seed if seed is not None else int(np.random.randint(0, 99999))
        self.all_jobs      = create_job_batch(self.n_jobs, seed=actual_seed, config=self.config)
        self.pending_jobs  = list(self.all_jobs)
        self.available_jobs = []
        self.completed_jobs = []
        self._rush_id_counter = 1000 + self.episode_id * 100

        # ── Reset disruption manager and energy price ─────────────
        self.disruption_mgr.reset()

        # ── Release any jobs arriving at step 0 ───────────────────
        self._release_pending_jobs()

        # ── Build observations ────────────────────────────────────
        observations = {
            agent_id: self._get_obs(i)
            for i, agent_id in enumerate(self.agents)
        }

        info = {
            'episode_id':   self.episode_id,
            'n_jobs':       self.n_jobs,
            'n_machines':   self.n_machines,
            'job_summary':  [repr(j) for j in self.all_jobs],
        }

        return observations, info

    # ──────────────────────────────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one time step.

        Args:
            actions: {agent_id: action_int}
                     action_int in [0, n_jobs] where n_jobs = WAIT

        Returns:
            observations  {agent_id: np.ndarray}
            rewards       {agent_id: float}
            terminateds   {agent_id: bool} + '__all__' key
            truncateds    {agent_id: bool} + '__all__' key
            infos         {agent_id: dict}
        """
        assert self.machines, "Call reset() before step()!"

        # Track step-level metrics for reward
        completed_this_step:    List[Job] = []
        tardiness_this_step:    float     = 0.0
        energy_this_step:       float     = 0.0
        disruption_recovery:    bool      = False

        # ══════════════════════════════════════════════════════════
        # 1. ADVANCE MACHINE PROCESSING
        # ══════════════════════════════════════════════════════════
        for machine in self.machines:
            if machine['status'] == 'busy' and machine['current_job'] is not None:
                job = machine['current_job']
                # Energy consumed this step (with spike multiplier)
                step_energy = job.energy_cost * self.disruption_mgr.get_energy_multiplier()
                energy_this_step    += step_energy
                self._episode_energy += step_energy

                # Decrement remaining steps
                machine['remaining_steps'] -= 1

                if machine['remaining_steps'] <= 0:
                    # ── Job completed ─────────────────────────────
                    job.completion_time = self.current_step
                    completed_this_step.append(job)
                    machine['completed_jobs'].append(job)
                    self.completed_jobs.append(job)
                    tardiness_this_step   += job.tardiness
                    machine['current_job']     = None
                    machine['status']          = 'idle'
                    machine['remaining_steps'] = 0

            elif machine['status'] == 'broken':
                # ── Count down repair ─────────────────────────────
                machine['repair_countdown'] -= 1
                if machine['repair_countdown'] <= 0:
                    machine['status']          = 'idle'
                    machine['repair_countdown']= 0

        # ══════════════════════════════════════════════════════════
        # 2. RELEASE NEWLY ARRIVED JOBS
        # ══════════════════════════════════════════════════════════
        self._release_pending_jobs()

        # ══════════════════════════════════════════════════════════
        # 3. EXECUTE AGENT ACTIONS
        # ══════════════════════════════════════════════════════════
        # Track which jobs are claimed this step to prevent double-assignment
        claimed_ids: set = set()

        for machine_idx, agent_id in enumerate(self.agents):
            action  = actions.get(agent_id, self.n_jobs)   # default WAIT
            machine = self.machines[machine_idx]

            # Only idle machines can take a new job
            if machine['status'] != 'idle':
                continue

            # WAIT action
            if action == self.n_jobs:
                continue

            # Out-of-range action — treat as WAIT
            if not (0 <= action < self.n_jobs):
                continue

            # Slot empty — no job at this position
            if action >= len(self.available_jobs):
                continue

            target_job = self.available_jobs[action]

            # Already claimed by another machine this step
            if target_job.job_id in claimed_ids:
                continue

            # ── Assign job to machine ─────────────────────────────
            target_job.start_time       = self.current_step
            target_job.assigned_machine = machine_idx
            machine['current_job']      = target_job
            machine['status']           = 'busy'
            machine['remaining_steps']  = target_job.processing_time
            claimed_ids.add(target_job.job_id)

        # Remove assigned jobs from the available pool
        self.available_jobs = [
            j for j in self.available_jobs if j.job_id not in claimed_ids
        ]
        # Re-sort after assignments to maintain urgency ordering.
        self.available_jobs.sort(key=lambda j: j.deadline)

        # ══════════════════════════════════════════════════════════
        # 4. SAMPLE AND APPLY DISRUPTIONS
        # ══════════════════════════════════════════════════════════
        events = self.disruption_mgr.sample_disruptions(
            self.current_step, self.machines
        )

        for event in events:

            if event.event_type == 'breakdown':
                mid = event.details['machine_id']
                m   = self.machines[mid]
                if m['status'] != 'broken':
                    # Pause and return current job to available queue
                    if m['current_job'] is not None:
                        paused = m['current_job']
                        paused.reset_assignment()
                        self.available_jobs.append(paused)
                        m['current_job'] = None
                    m['status']           = 'broken'
                    m['remaining_steps']  = 0
                    m['repair_countdown'] = DisruptionManager.REPAIR_STEPS
                    disruption_recovery   = len(completed_this_step) > 0

            elif event.event_type == 'rush_order':
                rush = create_rush_order(self.current_step, self._rush_id_counter)
                self._rush_id_counter += 1
                self.available_jobs.insert(0, rush)   # front of queue

            # energy_spike: already handled inside DisruptionManager

        # ══════════════════════════════════════════════════════════
        # 5. COMPUTE REWARD
        # ══════════════════════════════════════════════════════════
        n_idle = sum(1 for m in self.machines if m['status'] == 'idle')

        reward_val = self.reward_calc.compute(
            jobs_completed_this_step = completed_this_step,
            tardiness_this_step      = tardiness_this_step,
            energy_used_this_step    = energy_this_step,
            n_idle_machines          = n_idle,
            max_deadline             = self.max_steps,
            disruption_recovery      = disruption_recovery,
        )

        # ══════════════════════════════════════════════════════════
        # 6. ADVANCE CLOCK AND CHECK TERMINATION
        # ══════════════════════════════════════════════════════════
        self.current_step += 1

        all_jobs_done = (
            len(self.completed_jobs) >= len(self.all_jobs)
            and len(self.available_jobs) == 0
            and len(self.pending_jobs) == 0
        )
        time_exceeded = self.current_step >= self.max_steps

        terminated = all_jobs_done
        truncated  = time_exceeded and not all_jobs_done

        # ══════════════════════════════════════════════════════════
        # 7. BUILD OUTPUT DICTS
        # ══════════════════════════════════════════════════════════
        observations = {
            agent_id: self._get_obs(i)
            for i, agent_id in enumerate(self.agents)
        }

        rewards = {agent_id: reward_val for agent_id in self.agents}

        terminateds = {agent_id: terminated for agent_id in self.agents}
        terminateds['__all__'] = terminated

        truncateds = {agent_id: truncated for agent_id in self.agents}
        truncateds['__all__'] = truncated

        infos = {
            agent_id: {
                'step':            self.current_step,
                'completed':       len(self.completed_jobs),
                'available':       len(self.available_jobs),
                'pending':         len(self.pending_jobs),
                'energy_episode':  round(self._episode_energy, 3),
                'tardiness_rate':  self._compute_tardiness_rate(),
                'events':          [e.event_type for e in events],
            }
            for agent_id in self.agents
        }

        return observations, rewards, terminateds, truncateds, infos

    # ──────────────────────────────────────────────────────────────────
    #  OBSERVATION BUILDER
    # ──────────────────────────────────────────────────────────────────

    def _get_obs(self, machine_index: int) -> np.ndarray:
        """
        Build the 38-dimensional normalized observation vector for one agent.

        Layout:
            [0-8]   Machine status one-hot  (3 machines × 3 states)
            [9-11]  Normalized remaining processing steps per machine
            [12-35] Job queue features      (up to 6 jobs × 4 features)
            [36]    Normalized global clock
            [37]    Normalized current energy price
        """
        obs = np.zeros(38, dtype=np.float32)

        # ── [0-8] Machine status (one-hot per machine) ────────────
        # idle=index 0, busy=index 1, broken=index 2
        status_idx = {'idle': 0, 'busy': 1, 'broken': 2}
        for i, machine in enumerate(self.machines):
            si = status_idx.get(machine['status'], 0)
            obs[i * 3 + si] = 1.0

        # ── [9-11] Remaining processing time per machine ──────────
        for i, machine in enumerate(self.machines):
            if machine['status'] == 'busy' and machine['remaining_steps'] > 0:
                obs[9 + i] = min(
                    machine['remaining_steps'] / self.MAX_PROC_TIME, 1.0
                )

        # ── [12-35] Available job queue features ──────────────────
        # Pad with zeros if fewer than n_jobs are available
        for j in range(min(self.n_jobs, len(self.available_jobs))):
            job  = self.available_jobs[j]
            base = 12 + j * 4

            # Feature 0: Job type  (A=0.0, B=0.5, C=1.0)
            obs[base + 0] = job.type_to_float()

            # Feature 1: Processing time (normalized)
            obs[base + 1] = min(job.processing_time / self.MAX_PROC_TIME, 1.0)

            # Feature 2: Deadline urgency (remaining steps / max)
            remaining = max(0, job.deadline - self.current_step)
            obs[base + 2] = min(remaining / self.MAX_DEADLINE, 1.0)

            # Feature 3: Energy cost (normalized)
            obs[base + 3] = min(job.energy_cost / self.MAX_ENERGY_COST, 1.0)

        # ── [36] Global clock ─────────────────────────────────────
        obs[36] = self.current_step / self.max_steps

        # ── [37] Energy price ─────────────────────────────────────
        obs[37] = self.disruption_mgr.get_energy_multiplier() / self.MAX_SPIKE_FACTOR

        return obs

    # ──────────────────────────────────────────────────────────────────
    #  ACTION MASK
    # ──────────────────────────────────────────────────────────────────

    def get_action_mask(self, machine_index: int) -> np.ndarray:
        """
        Boolean mask for valid actions for one machine.

        True  = valid action
        False = invalid (empty slot or machine not idle)
        """
        mask    = np.zeros(self.n_jobs + 1, dtype=bool)
        machine = self.machines[machine_index]

        # WAIT is always valid
        mask[self.n_jobs] = True

        # Only idle machines can select a job
        if machine['status'] == 'idle':
            for j in range(min(self.n_jobs, len(self.available_jobs))):
                mask[j] = True

        return mask

    # ──────────────────────────────────────────────────────────────────
    #  RENDER
    # ──────────────────────────────────────────────────────────────────

    def render(self) -> dict:
        """
        Return full factory state as a dict for the Streamlit dashboard.
        """
        return {
            'step':             self.current_step,
            'max_steps':        self.max_steps,
            'energy_price':     self.disruption_mgr.get_energy_multiplier(),
            'spike_active':     self.disruption_mgr.is_spike_active(),
            'machines': [
                {
                    'id':           m['id'],
                    'status':       m['status'],
                    'current_job':  m['current_job'].job_id if m['current_job'] else None,
                    'remaining':    m['remaining_steps'],
                    'repair':       m['repair_countdown'],
                    'n_completed':  len(m['completed_jobs']),
                }
                for m in self.machines
            ],
            'available_jobs': [
                {
                    'id':       j.job_id,
                    'type':     j.job_type,
                    'proc':     j.processing_time,
                    'deadline': j.deadline,
                    'energy':   j.energy_cost,
                    'priority': j.priority,
                }
                for j in self.available_jobs
            ],
            'gantt_jobs': [
                {
                    'id':       j.job_id,
                    'type':     j.job_type,
                    'machine':  j.assigned_machine,
                    'start':    j.start_time,
                    'duration': j.processing_time,
                    'deadline': j.deadline,
                    'tardiness':j.tardiness,
                    'priority': j.priority,
                }
                for j in self.completed_jobs
                if j.start_time is not None
            ],
            'metrics': {
                'completed':       len(self.completed_jobs),
                'total':           len(self.all_jobs),
                'tardiness_rate':  round(self._compute_tardiness_rate(), 3),
                'episode_energy':  round(self._episode_energy, 3),
                'breakdowns':      self.disruption_mgr.get_breakdown_count(),
                'rush_orders':     self.disruption_mgr.get_rush_order_count(),
            },
        }

    # ──────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _release_pending_jobs(self):
        """Move jobs whose arrival_time <= current_step into available_jobs."""
        newly_arrived = []
        still_pending = []
        for job in self.pending_jobs:
            if job.arrival_time <= self.current_step:
                newly_arrived.append(job)
            else:
                still_pending.append(job)
        self.pending_jobs   = still_pending
        self.available_jobs.extend(newly_arrived)
        # Sort by deadline — most urgent (smallest deadline) first.
        # Action 0 = most urgent job. Canonicalizes the action space.
        self.available_jobs.sort(key=lambda j: j.deadline)

    def _compute_tardiness_rate(self) -> float:
        """Fraction of completed jobs that missed their deadline."""
        if not self.completed_jobs:
            return 0.0
        tardy = sum(1 for j in self.completed_jobs if j.tardiness > 0)
        return tardy / len(self.completed_jobs)

    def close(self):
        pass

    def __repr__(self):
        return (
            f"FactoryGym(machines={self.n_machines}, jobs={self.n_jobs}, "
            f"max_steps={self.max_steps}, step={self.current_step})"
        )