from env.job_generator import JOB_TYPE_CONFIG, create_job_batch
from env.factory_gym import FactoryGym
import json, numpy as np

# Verify config
with open('config.json') as f:
    cfg = json.load(f)
assert cfg['breakdown_rate'] == 0.003, f'Wrong breakdown_rate: {cfg["breakdown_rate"]}'
print(f'  config.json OK  breakdown_rate={cfg["breakdown_rate"]}')

# Verify slacks
assert JOB_TYPE_CONFIG['A']['deadline_slack'] == 5
assert JOB_TYPE_CONFIG['B']['deadline_slack'] == 8
assert JOB_TYPE_CONFIG['C']['deadline_slack'] == 12
print('  job_generator.py OK  slacks A=5, B=8, C=12')

# Verify arrival window
jobs = create_job_batch(6, seed=42, config={'max_steps': 100})
max_arr = max(j.arrival_time for j in jobs)
assert max_arr <= 5, f'Jobs arriving at step {max_arr}, expected <= 5'
print(f'  Arrival window OK  max arrival step = {max_arr}')

# Verify deadline sorting
env = FactoryGym({'n_machines':3,'n_jobs':6,'max_steps':100,'breakdown_rate':0.003,'rush_rate':0.005,'energy_spike_rate':0.05})
obs, _ = env.reset(seed=42)
actions = {aid: env.n_jobs for aid in env.agents}
for _ in range(5):
    obs, _, t, tr, _ = env.step(actions)
    if env.available_jobs:
        dls = [j.deadline for j in env.available_jobs]
        assert dls == sorted(dls), f'Sort broken! {dls}'
    if t['__all__'] or tr['__all__']:
        break
print('  factory_gym.py OK  deadline sorting maintained')
print()
print('ALL LOCAL FILES MATCH TRAINING CONFIG')
