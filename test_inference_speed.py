# Save as test_inference_speed.py
import time
import numpy as np
from env.factory_gym import FactoryGym
from agents.mappo_agent import MAPPOAgent

env   = FactoryGym()
agent = MAPPOAgent()
agent.load('models/mappo_factory_final.pth')
agent.set_eval_mode()

obs, _ = env.reset(seed=0)

# Warm up
for _ in range(10):
    masks = {f'machine_{i}': env.get_action_mask(i) for i in range(env.n_machines)}
    agent.predict(obs, masks)

# Timed run
n_timed  = 500
times    = []

for _ in range(n_timed):
    masks = {f'machine_{i}': env.get_action_mask(i) for i in range(env.n_machines)}
    t0 = time.perf_counter()
    actions, _, _ = agent.predict(obs, masks)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)   # ms
    obs, _, terms, truncs, _ = env.step(actions)
    if terms['__all__'] or truncs['__all__']:
        obs, _ = env.reset()

mean_ms = np.mean(times)
p95_ms  = np.percentile(times, 95)
max_ms  = np.max(times)

print(f"Inference Speed Test  ({n_timed} steps)")
print(f"  Mean latency : {mean_ms:.2f} ms")
print(f"  95th pct     : {p95_ms:.2f} ms")
print(f"  Max latency  : {max_ms:.2f} ms")
print(f"  Target       : < 50 ms per step")

if mean_ms < 50:
    print(f"\nSPEED TEST PASSED — {mean_ms:.1f}ms mean (target < 50ms)")
else:
    print(f"\nWARNING — {mean_ms:.1f}ms mean exceeds 50ms target")
    print("This may be acceptable on slower machines — check on demo laptop specifically")