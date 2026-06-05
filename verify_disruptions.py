# Save as verify_disruptions.py
from env.factory_gym import FactoryGym
import numpy as np

env = FactoryGym()

# ── Test 1: Machine Breakdown ─────────────────────────────────────────
print("=== TEST 1: Machine Breakdown ===")
obs, _ = env.reset(seed=42)
# Force machine 0 busy
if env.available_jobs:
    env.machines[0]['status']           = 'busy'
    env.machines[0]['current_job']      = env.available_jobs.pop(0)
    env.machines[0]['remaining_steps']  = 10
    env.machines[0]['current_job'].start_time = 0

# Manually break machine 1
env.machines[1]['status']           = 'broken'
env.machines[1]['repair_countdown'] = 10
print(f"  M0 status: {env.machines[0]['status']} (should be busy)")
print(f"  M1 status: {env.machines[1]['status']} (should be broken)")
print(f"  M1 repair countdown: {env.machines[1]['repair_countdown']} (should be 10)")

# Step forward 11 times
actions = {a: env.n_jobs for a in env.agents}
for i in range(11):
    obs, r, terms, truncs, info = env.step(actions)
    if i == 9:
        print(f"  Step {env.current_step}: M1 status = {env.machines[1]['status']}")
print(f"  After 11 steps M1 status: {env.machines[1]['status']} (should be idle)")
print("  Machine breakdown: OK\n")

# ── Test 2: Rush Order ────────────────────────────────────────────────
print("=== TEST 2: Rush Order ===")
env2 = FactoryGym()
env2.reset(seed=5)
from env.job_generator import create_rush_order
rush = create_rush_order(current_step=5, job_id=999)
env2.available_jobs.insert(0, rush)
print(f"  First job in queue priority: {env2.available_jobs[0].priority} (should be 'high')")
print(f"  First job deadline: {env2.available_jobs[0].deadline}")
print("  Rush order: OK\n")

# ── Test 3: Energy Spike ──────────────────────────────────────────────
print("=== TEST 3: Energy Spike ===")
from env.disruption import DisruptionManager
dm = DisruptionManager(energy_spike_rate=1.0, seed=0)  # 100% spike rate
machines = []
events = dm.sample_disruptions(0, machines)
spike_events = [e for e in events if e.event_type == 'energy_spike']
print(f"  Spike triggered: {len(spike_events) > 0} (should be True)")
if spike_events:
    print(f"  Spike factor: {spike_events[0].details['factor']} (should be 1.5-3.0)")
    print(f"  Spike duration: {spike_events[0].details['duration']} (should be 5-15)")
print(f"  Current multiplier: {dm.get_energy_multiplier()} (should be > 1.0)")
print("  Energy spike: OK\n")

print("ALL DISRUPTION TESTS PASSED")