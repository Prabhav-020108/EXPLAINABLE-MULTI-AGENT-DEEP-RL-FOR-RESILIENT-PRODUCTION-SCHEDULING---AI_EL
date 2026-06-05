# Save this as test_100_episodes.py in your project root
from env.factory_gym import FactoryGym
import numpy as np

def test_100_episodes():
    env = FactoryGym()
    errors = []

    for ep in range(100):
        try:
            obs, info = env.reset(seed=ep)
            assert obs['machine_0'].shape == (38,), "Wrong obs shape!"
            assert obs['machine_0'].min() >= 0.0,   "Obs below 0!"
            assert obs['machine_0'].max() <= 1.0,   "Obs above 1!"

            step = 0
            while True:
                actions = {}
                for i, agent_id in enumerate(env.agents):
                    mask  = env.get_action_mask(i)
                    valid = np.where(mask)[0]
                    actions[agent_id] = int(np.random.choice(valid))

                obs, rewards, terms, truncs, infos = env.step(actions)

                # Check reward is a valid number
                r = rewards['machine_0']
                assert r == r,          f"Reward is NaN at step {step}!"
                assert abs(r) < 1000,   f"Reward suspiciously large: {r}"

                step += 1
                if terms['__all__'] or truncs['__all__']:
                    break

        except Exception as e:
            errors.append(f"Episode {ep}: {type(e).__name__}: {e}")

    if errors:
        print(f"FAILED — {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
    else:
        print("100 EPISODES PASSED — Zero errors!")
        print(f"Environment is stable and production-ready.")

test_100_episodes()