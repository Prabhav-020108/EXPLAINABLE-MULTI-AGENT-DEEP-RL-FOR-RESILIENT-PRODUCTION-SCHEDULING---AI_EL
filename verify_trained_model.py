# Save as verify_trained_model.py
import numpy as np
from env.factory_gym import FactoryGym
from agents.mappo_agent import MAPPOAgent

env   = FactoryGym()
agent = MAPPOAgent()
agent.load('models/mappo_factory_final.pth')
agent.set_eval_mode()

tardinesses = []
makespans   = []
energies    = []

print("Running 20 evaluation episodes with trained model...")
print("-" * 50)

for ep in range(20):
    obs, _ = env.reset(seed=ep)
    step   = 0

    while True:
        masks   = {f'machine_{i}': env.get_action_mask(i) for i in range(env.n_machines)}
        actions, _, _ = agent.predict(obs, masks)
        obs, _, terms, truncs, _ = env.step(actions)
        step += 1
        if terms['__all__'] or truncs['__all__']:
            break

    state = env.render()
    t = state['metrics']['tardiness_rate']
    e = state['metrics']['episode_energy']
    tardinesses.append(t)
    makespans.append(step)
    energies.append(e)
    print(f"  Ep {ep+1:>2}: makespan={step:>3}  tardiness={t:.1%}  energy={e:.2f}")

print("-" * 50)
print(f"  Mean makespan : {np.mean(makespans):.1f}")
print(f"  Mean tardiness: {np.mean(tardinesses):.1%}  (target: < 15%)")
print(f"  Mean energy   : {np.mean(energies):.2f}")
print()

if np.mean(tardinesses) < 0.15:
    print("TRAINED MODEL VERIFIED — Tardiness target met!")
elif np.mean(tardinesses) < 0.25:
    print("TRAINED MODEL OK — Tardiness above 15% but within acceptable range.")
    print("Consider longer training (1.5M steps) for better performance.")
else:
    print("WARNING — Tardiness too high. Model may need more training.")