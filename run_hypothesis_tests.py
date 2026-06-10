# Save as run_hypothesis_tests.py in project root
import json
import numpy as np
from agents.mappo_agent    import MAPPOAgent
from xai.shap_explainer    import SHAPExplainer
from xai.hypothesis_tester import HypothesisTester

ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
}

print("Loading trained agent...")
agent = MAPPOAgent()
agent.load('models/mappo_factory_best.pth')
agent.set_eval_mode()

print("Building SHAP explainer (100 background states — takes ~60 seconds)...")
explainer = SHAPExplainer(
    agent.actors[0],
    ENV_CONFIG,
    n_background=100,
    verbose=True,
)

print("Running all hypothesis tests...")
tester  = HypothesisTester(explainer, verbose=True)
results = tester.run_all()

# Print final verdict
print()
print("=" * 60)
print("FINAL HYPOTHESIS VERIFICATION REPORT")
print("=" * 60)
for h_id, r in results.items():
    status = "✅ VERIFIED" if r['result'] == 'VERIFIED' else "❌ NOT VERIFIED"
    print(f"  {h_id}: {status}  —  {r['description']}")
print()

n_verified = sum(1 for r in results.values() if r['result'] == 'VERIFIED')
print(f"  {n_verified}/3 hypotheses verified")
print()
print("Results saved to: logs/hypothesis_results.json")