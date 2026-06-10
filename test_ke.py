import numpy as np
import torch
import json
from agents.mappo_agent import MAPPOAgent
from xai.shap_explainer import SHAPExplainer, _LogitActor
from env.factory_gym import FactoryGym

ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
}

agent = MAPPOAgent()
agent.load('models/mappo_factory_best.pth')
agent.set_eval_mode()

import shap

def run_kernel():
    explainer = SHAPExplainer(agent.actors[0], ENV_CONFIG, n_background=50, verbose=False)
    bg = explainer.background_np
    
    # We want to explain logits
    model = _LogitActor(agent.actors[0])
    model.eval()
    def f(x):
        with torch.no_grad():
            return model(torch.FloatTensor(x)).numpy()
            
    ke = shap.KernelExplainer(f, bg[:10])
    
    # Let's create H1 observation
    obs = np.zeros(38, dtype=np.float32)
    obs[0] = 1.0; obs[4] = 1.0; obs[7] = 1.0
    obs[10] = 0.40; obs[11] = 0.35
    obs[36] = 0.3; obs[37] = 0.05
    
    obs[12] = 0.0; obs[13] = 0.15; obs[14] = 0.001; obs[15] = 0.15
    for slot in range(1, 6):
        base = 12 + slot * 4
        obs[base] = 0.5; obs[base+1] = 0.40; obs[base+2] = 0.60; obs[base+3] = 0.45
        
    shap_values = ke.shap_values(obs, nsamples=100, silent=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0] # For action 0
    
    print("KernelExplainer SHAP for Action 0:")
    total = np.sum(np.abs(shap_values))
    print(f"Total sum abs: {total}")
    print(f"obs[14] pct: {abs(shap_values[14])/total*100:.1f}%")
    print(f"obs[37] pct: {abs(shap_values[37])/total*100:.1f}%")

run_kernel()
