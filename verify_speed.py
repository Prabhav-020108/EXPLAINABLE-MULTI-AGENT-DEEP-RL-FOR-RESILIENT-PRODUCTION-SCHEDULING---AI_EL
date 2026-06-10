import time, numpy as np
from agents.mappo_agent import MAPPOAgent
from xai.shap_explainer import SHAPExplainer
from env.factory_gym import FactoryGym

ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
}

agent = MAPPOAgent()
agent.load('models/mappo_factory_best.pth')
agent.set_eval_mode()

explainer = SHAPExplainer(agent.actors[0], ENV_CONFIG, n_background=50, verbose=False)

env = FactoryGym(ENV_CONFIG)
times = []
for ep in range(5):
    obs, _ = env.reset(seed=ep)
    t0 = time.time()
    result = explainer.explain(obs['machine_0'])
    t1 = time.time()
    times.append((t1 - t0) * 1000)
    print(f'  Ep {ep+1}: {times[-1]:.0f}ms  action={result["chosen_action"]}  '
          f'top={result["top_features"][0][0]} ({result["top_features"][0][1]:.1f}%)')

print()
print(f'Mean explanation time : {np.mean(times):.0f}ms')
print(f'Max  explanation time : {max(times):.0f}ms')
print(f'Target                : < 500ms')
if np.mean(times) < 500:
    print('SPEED TEST PASSED')
else:
    print('Too slow — reduce n_background to 30')
