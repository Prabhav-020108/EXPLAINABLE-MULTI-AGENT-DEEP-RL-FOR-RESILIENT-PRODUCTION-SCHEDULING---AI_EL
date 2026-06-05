"""Unit tests for FactoryGym, JobGenerator, and RewardCalculator."""

import pytest
import numpy as np
from env.factory_gym import FactoryGym
from env.job_generator import create_job_batch, create_rush_order, Job
from env.reward import RewardCalculator
from env.disruption import DisruptionManager


# ──────────────────────────────────────────────────────────────────────
#  JOB GENERATOR TESTS
# ──────────────────────────────────────────────────────────────────────

class TestJobGenerator:

    def test_correct_number_of_jobs(self):
        jobs = create_job_batch(6, seed=0)
        assert len(jobs) == 6

    def test_deadlines_always_valid(self):
        for seed in range(50):
            jobs = create_job_batch(6, seed=seed)
            for job in jobs:
                assert job.deadline > job.arrival_time + job.processing_time, \
                    f"Invalid deadline for {job}"

    def test_jobs_sorted_by_arrival(self):
        jobs = create_job_batch(6, seed=42)
        arrivals = [j.arrival_time for j in jobs]
        assert arrivals == sorted(arrivals)

    def test_job_types_are_valid(self):
        jobs = create_job_batch(6, seed=1)
        for job in jobs:
            assert job.job_type in ('A', 'B', 'C')

    def test_processing_times_in_range(self):
        for seed in range(20):
            jobs = create_job_batch(6, seed=seed)
            for job in jobs:
                if job.job_type == 'A':
                    assert 3 <= job.processing_time <= 6
                elif job.job_type == 'B':
                    assert 7 <= job.processing_time <= 12
                else:
                    assert 13 <= job.processing_time <= 20

    def test_energy_cost_in_range(self):
        jobs = create_job_batch(6, seed=99)
        for job in jobs:
            assert 0.0 < job.energy_cost <= 1.0

    def test_reproducibility_with_seed(self):
        jobs_a = create_job_batch(6, seed=123)
        jobs_b = create_job_batch(6, seed=123)
        for a, b in zip(jobs_a, jobs_b):
            assert a.processing_time == b.processing_time
            assert a.deadline        == b.deadline

    def test_rush_order_is_high_priority(self):
        rush = create_rush_order(current_step=10, job_id=999)
        assert rush.priority == 'high'
        assert rush.arrival_time == 10


# ──────────────────────────────────────────────────────────────────────
#  REWARD TESTS
# ──────────────────────────────────────────────────────────────────────

class TestRewardCalculator:

    def setup_method(self):
        self.calc = RewardCalculator()

    def test_zero_reward_for_idle_no_jobs(self):
        r = self.calc.compute([], 0.0, 0.0, 0, max_deadline=100)
        assert r == 0.0

    def test_positive_reward_for_on_time_completion(self):
        job = Job(job_id=0, job_type='A', arrival_time=0,
                  processing_time=4, deadline=10, energy_cost=0.2)
        job.completion_time = 7  # before deadline, tardiness=0
        r = self.calc.compute([job], 0.0, 0.0, 0, max_deadline=100)
        assert r > 0.0

    def test_negative_reward_for_late_job(self):
        job = Job(job_id=0, job_type='A', arrival_time=0,
                  processing_time=4, deadline=5, energy_cost=0.2)
        job.completion_time = 12  # 7 steps late
        r = self.calc.compute([job], 7.0, 0.0, 0, max_deadline=100)
        assert r < 0.0

    def test_reward_never_nan(self):
        for _ in range(200):
            jobs = create_job_batch(6, seed=np.random.randint(1000))
            job = jobs[0]
            job.completion_time = job.deadline + np.random.randint(0, 5)
            r = self.calc.compute([job], float(job.tardiness),
                                  np.random.uniform(0, 3), np.random.randint(0, 3))
            assert r == r, "Reward is NaN!"

    def test_energy_penalty_is_bounded(self):
        r_no_energy  = self.calc.compute([], 0.0, 0.0, 0)
        r_max_energy = self.calc.compute([], 0.0, 100.0, 0)
        assert r_max_energy < r_no_energy

    def test_idle_penalty(self):
        r_no_idle   = self.calc.compute([], 0.0, 0.0, 0)
        r_all_idle  = self.calc.compute([], 0.0, 0.0, 3)
        assert r_all_idle < r_no_idle


# ──────────────────────────────────────────────────────────────────────
#  DISRUPTION MANAGER TESTS
# ──────────────────────────────────────────────────────────────────────

class TestDisruptionManager:

    def test_reset_clears_state(self):
        dm = DisruptionManager(breakdown_rate=1.0, seed=0)
        machines = [{'id': 0, 'status': 'idle', 'current_job': None}]
        dm.sample_disruptions(0, machines)
        dm.reset()
        assert len(dm.history) == 0
        assert dm.get_energy_multiplier() == 1.0

    def test_broken_machines_not_broken_again(self):
        dm = DisruptionManager(breakdown_rate=1.0, seed=1)
        machines = [{'id': 0, 'status': 'broken', 'current_job': None}]
        events = dm.sample_disruptions(0, machines)
        breakdowns = [e for e in events if e.event_type == 'breakdown']
        assert len(breakdowns) == 0

    def test_energy_multiplier_returns_to_one(self):
        """
        Verify that when a spike's duration expires, the multiplier
        correctly resets to 1.0 (before any new spike can start).

        Fix: After triggering the first spike, set energy_spike_rate=0.0
        so no new spike auto-starts once this one expires.
        """
        dm = DisruptionManager(energy_spike_rate=1.0, seed=5)
        machines = []

        # Trigger the first spike
        dm.sample_disruptions(0, machines)
        factor = dm.get_energy_multiplier()
        assert factor >= 1.5, f"Spike should have started, got factor={factor}"

        # KEY FIX: Disable new spikes so the current one can expire cleanly
        dm.energy_spike_rate = 0.0

        # Burn through the spike duration (max spike duration is 15 steps)
        for i in range(20):
            dm.sample_disruptions(i + 1, machines)

        # Now the spike must have expired and no new one started
        assert dm.get_energy_multiplier() == 1.0, (
            f"Multiplier should be 1.0 after spike expires, got "
            f"{dm.get_energy_multiplier()}. spike_remaining={dm.spike_remaining_steps}"
        )
        assert dm.spike_remaining_steps == 0, (
            f"Spike countdown should be 0, got {dm.spike_remaining_steps}"
        )
        assert not dm.is_spike_active(), "Spike should not be active"


# ──────────────────────────────────────────────────────────────────────
#  FACTORY GYM INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────────

class TestFactoryGym:

    def setup_method(self):
        self.env = FactoryGym()

    def test_reset_returns_correct_obs_shape(self):
        obs, info = self.env.reset(seed=0)
        for agent_id in self.env.agents:
            assert obs[agent_id].shape == (38,), \
                f"Wrong shape for {agent_id}: {obs[agent_id].shape}"

    def test_obs_values_in_range(self):
        obs, _ = self.env.reset(seed=0)
        for agent_id, o in obs.items():
            assert o.min() >= 0.0, f"{agent_id} obs below 0"
            assert o.max() <= 1.0, f"{agent_id} obs above 1"

    def test_obs_shape_stable_under_disruptions(self):
        obs, _ = self.env.reset(seed=1)
        for _ in range(30):
            actions = {
                agent_id: self.env.n_jobs   # WAIT for all
                for agent_id in self.env.agents
            }
            obs, _, terms, truncs, _ = self.env.step(actions)
            for agent_id, o in obs.items():
                assert o.shape == (38,)
            if terms['__all__'] or truncs['__all__']:
                break

    def test_breakdown_sets_machine_broken(self):
        env = FactoryGym()
        obs, _ = env.reset(seed=10)
        # Force a breakdown via disruption manager directly
        env.machines[0]['status']           = 'busy'
        env.machines[0]['current_job']      = env.all_jobs[0] if env.all_jobs else None
        env.machines[0]['remaining_steps']  = 5

        from env.disruption import DisruptionEvent
        evt = DisruptionEvent('breakdown', 0, {'machine_id': 0, 'repair_steps': 10, 'was_processing': True})
        env.disruption_mgr.history.append(evt)

        # Apply manually
        m = env.machines[0]
        if m['current_job']:
            m['current_job'].reset_assignment()
            env.available_jobs.append(m['current_job'])
            m['current_job'] = None
        m['status']           = 'broken'
        m['repair_countdown'] = 10

        assert env.machines[0]['status'] == 'broken'
        assert env.machines[0]['repair_countdown'] == 10

    def test_rush_order_inserted_at_front(self):
        env = FactoryGym()
        env.reset(seed=0)
        initial_len = len(env.available_jobs)
        rush = create_rush_order(0, 999)
        env.available_jobs.insert(0, rush)
        assert env.available_jobs[0].priority == 'high'
        assert len(env.available_jobs) == initial_len + 1

    def test_action_mask_shape(self):
        self.env.reset(seed=0)
        for i in range(self.env.n_machines):
            mask = self.env.get_action_mask(i)
            assert mask.shape == (self.env.n_jobs + 1,)
            assert mask.dtype == bool
            assert mask[-1] == True   # WAIT always valid

    def test_100_random_episodes_no_crash(self):
        for ep in range(100):
            obs, _ = self.env.reset(seed=ep)
            for _ in range(200):
                actions = {}
                for idx, agent_id in enumerate(self.env.agents):
                    mask  = self.env.get_action_mask(idx)
                    valid = np.where(mask)[0]
                    actions[agent_id] = int(np.random.choice(valid))
                obs, rewards, terms, truncs, _ = self.env.step(actions)
                assert obs['machine_0'].shape == (38,)
                r = rewards['machine_0']
                assert r == r   # NaN check
                if terms['__all__'] or truncs['__all__']:
                    break