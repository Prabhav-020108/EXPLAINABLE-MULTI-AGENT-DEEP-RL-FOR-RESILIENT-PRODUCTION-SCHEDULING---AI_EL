import numpy as np
from agents.mappo_agent import MAPPOAgent
from xai.shap_explainer import SHAPExplainer, FEATURE_NAMES, FEATURE_GROUPS

# Load trained agent
agent = MAPPOAgent()
agent.load('models/mappo_factory_best.pth')
agent.set_eval_mode()

ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
}

# Initialize explainer (this will take 30-60 seconds for background collection)
explainer = SHAPExplainer(agent.actors[0], ENV_CONFIG, n_background=20)

# Run one explanation with a dummy observation
from env.factory_gym import FactoryGym
env = FactoryGym(ENV_CONFIG)
obs, _ = env.reset(seed=7)

print('Running SHAP explanation...')
import time
t0 = time.time()
result = explainer.explain(obs['machine_0'])
t1 = time.time()

print(f'Explanation time: {(t1-t0)*1000:.0f}ms')
print()
print(result['narrative'])
print()
print(f'chosen_action : {result["chosen_action"]}')
print(f'action_prob   : {result["action_prob"]:.3f}')
print(f'shap_values   : shape={result["shap_values"].shape}')
print(f'top_features  : {result["top_features"][:3]}')
print(f'groups        : {result["groups"]}')

# Basic assertions
assert result['chosen_action'] in range(7)
assert 0.0 <= result['action_prob'] <= 1.0
assert result['shap_values'].shape == (38,)
assert len(result['top_features']) == 6
assert set(result['groups'].keys()) == set(FEATURE_GROUPS.keys())
assert len(FEATURE_NAMES) == 38

print()
print('shap_explainer.py OK — All assertions passed')
