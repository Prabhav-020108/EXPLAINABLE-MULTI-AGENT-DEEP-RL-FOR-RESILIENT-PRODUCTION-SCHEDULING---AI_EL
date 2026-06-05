"""Additional reward function tests."""

import numpy as np
import pytest
from env.reward import RewardCalculator
from env.job_generator import Job


def make_job(job_id=0, job_type='A', arr=0, proc=5, deadline=10,
             energy=0.2, completion=None):
    j = Job(job_id=job_id, job_type=job_type, arrival_time=arr,
            processing_time=proc, deadline=deadline, energy_cost=energy)
    j.completion_time = completion
    return j


def test_weight_sum_must_equal_one():
    calc = RewardCalculator()
    total = calc.w_completion + calc.w_tardiness + calc.w_energy + calc.w_idle
    assert abs(total - 1.0) < 1e-6


def test_late_job_penalizes_more_than_early():
    calc = RewardCalculator()
    j_early = make_job(completion=8)   # 2 steps early, no tardiness
    j_late  = make_job(completion=15)  # 5 steps late

    r_early = calc.compute([j_early], 0.0, 0.0, 0, max_deadline=100)
    r_late  = calc.compute([j_late],  5.0, 0.0, 0, max_deadline=100)
    assert r_early > r_late


def test_high_energy_penalizes_more_than_low():
    calc = RewardCalculator()
    r_low  = calc.compute([], 0.0, 0.1, 1)
    r_high = calc.compute([], 0.0, 5.0, 1)
    assert r_low > r_high


def test_reward_bounded_for_extreme_inputs():
    calc = RewardCalculator()
    r = calc.compute([], 1000.0, 1000.0, 10, max_deadline=100)
    assert abs(r) < 1e6   # Should not be astronomically large
    assert r == r          # Not NaN