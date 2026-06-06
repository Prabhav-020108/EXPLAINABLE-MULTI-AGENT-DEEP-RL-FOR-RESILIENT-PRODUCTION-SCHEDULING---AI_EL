# Save as verify_best_model.py in project root
import numpy as np
from env.factory_gym import FactoryGym
from agents.mappo_agent import MAPPOAgent

env   = FactoryGym()
agent = MAPPOAgent()

# ── Test BEST model ───────────────────────────────────────────────
print("="*55)
print("Testing BEST model (mappo_factory_best.pth)")
print("="*55)
agent.load('models/mappo_factory_best.pth')
agent.set_eval_mode()

tardinesses = []
makespans   = []
energies    = []
rewards     = []

for ep in range(50):   # 50 episodes for stable mean
    obs, _ = env.reset(seed=ep)
    ep_reward = 0.0
    step = 0
    while True:
        masks = {f'machine_{i}': env.get_action_mask(i)
                 for i in range(env.n_machines)}
        actions, _, _ = agent.predict(obs, masks)
        obs, rews, terms, truncs, _ = env.step(actions)
        ep_reward += rews['machine_0']
        step += 1
        if terms['__all__'] or truncs['__all__']:
            break
    state = env.render()
    tardinesses.append(state['metrics']['tardiness_rate'])
    makespans.append(step)
    energies.append(state['metrics']['episode_energy'])
    rewards.append(ep_reward)

print(f"\n  Episodes evaluated : 50")
print(f"  Mean reward        : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  Mean tardiness     : {np.mean(tardinesses):.1%} ± {np.std(tardinesses):.1%}")
print(f"  Min tardiness      : {np.min(tardinesses):.1%}")
print(f"  Max tardiness      : {np.max(tardinesses):.1%}")
print(f"  Mean makespan      : {np.mean(makespans):.1f}")
print(f"  Mean energy        : {np.mean(energies):.2f}")

print()
target_met = np.mean(tardinesses) < 0.15
print(f"  Tardiness target (< 15%): {'✅ MET' if target_met else '❌ NOT MET'}")

# ── Test FINAL model for comparison ──────────────────────────────
print()
print("="*55)
print("Testing FINAL model (mappo_factory_final.pth)")
print("="*55)
agent2 = MAPPOAgent()
agent2.load('models/mappo_factory_final.pth')
agent2.set_eval_mode()

tard2 = []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    step = 0
    while True:
        masks = {f'machine_{i}': env.get_action_mask(i)
                 for i in range(env.n_machines)}
        actions, _, _ = agent2.predict(obs, masks)
        obs, _, terms, truncs, _ = env.step(actions)
        step += 1
        if terms['__all__'] or truncs['__all__']:
            break
    state = env.render()
    tard2.append(state['metrics']['tardiness_rate'])

print(f"  Mean tardiness : {np.mean(tard2):.1%}")
print()
print(f"  Recommendation: Use {'BEST' if np.mean(tardinesses) < np.mean(tard2) else 'FINAL'} model")