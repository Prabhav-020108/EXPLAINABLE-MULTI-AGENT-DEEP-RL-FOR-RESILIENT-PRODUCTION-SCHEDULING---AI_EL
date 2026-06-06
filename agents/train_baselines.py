"""
train_baselines.py
------------------
Classical scheduling baselines (FCFS, SPT, EDD, LPT) for comparison
with our trained MAPPO agents.

Usage:
    python agents/train_baselines.py --episodes 100
    python agents/train_baselines.py --episodes 100 --methods FCFS SPT EDD
    python agents/train_baselines.py --episodes 100 --model-path models/
"""

import argparse
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.factory_gym import FactoryGym


# ════════════════════════════════════════════════════════════════════
#  BASELINE DISPATCHING RULES
# ════════════════════════════════════════════════════════════════════

class FCFSBaseline:
    """First Come First Served — pick job with earliest arrival_time."""
    name = 'FCFS'

    def decide(self, machine_idx, available_jobs, current_step, n_jobs):
        if not available_jobs:
            return n_jobs   # WAIT
        return min(range(len(available_jobs)),
                   key=lambda j: available_jobs[j].arrival_time)


class SPTBaseline:
    """Shortest Processing Time — pick job with smallest processing_time."""
    name = 'SPT'

    def decide(self, machine_idx, available_jobs, current_step, n_jobs):
        if not available_jobs:
            return n_jobs
        return min(range(len(available_jobs)),
                   key=lambda j: available_jobs[j].processing_time)


class EDDBaseline:
    """Earliest Due Date — pick job whose deadline is soonest."""
    name = 'EDD'

    def decide(self, machine_idx, available_jobs, current_step, n_jobs):
        if not available_jobs:
            return n_jobs
        return min(range(len(available_jobs)),
                   key=lambda j: available_jobs[j].deadline)


class LPTBaseline:
    """Longest Processing Time — pick job with largest processing_time."""
    name = 'LPT'

    def decide(self, machine_idx, available_jobs, current_step, n_jobs):
        if not available_jobs:
            return n_jobs
        return max(range(len(available_jobs)),
                   key=lambda j: available_jobs[j].processing_time)


class RandomBaseline:
    """Random — select a random valid job. Used as a lower bound."""
    name = 'Random'

    def decide(self, machine_idx, available_jobs, current_step, n_jobs):
        if not available_jobs:
            return n_jobs
        return int(np.random.randint(0, len(available_jobs)))


BASELINES = {
    'FCFS':   FCFSBaseline,
    'SPT':    SPTBaseline,
    'EDD':    EDDBaseline,
    'LPT':    LPTBaseline,
    'Random': RandomBaseline,
}


# ════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ════════════════════════════════════════════════════════════════════

def run_one_baseline(baseline_cls, env: FactoryGym, n_episodes: int,
                     seed_offset: int = 0) -> dict:
    """
    Run a baseline policy for n_episodes and collect metrics.

    Args:
        baseline_cls: The dispatching rule class (not instance).
        env:          FactoryGym instance.
        n_episodes:   Number of evaluation episodes.
        seed_offset:  Random seed start.

    Returns:
        Dictionary with mean and std for each metric.
    """
    baseline   = baseline_cls()
    makespans  = []
    tardinesses= []
    energies   = []
    completions= []
    rewards    = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        step_reward = 0.0
        step = 0

        while True:
            actions = {}
            for i, agent_id in enumerate(env.agents):
                machine = env.machines[i]

                if machine['status'] != 'idle':
                    # Machine busy or broken — must WAIT
                    actions[agent_id] = env.n_jobs
                else:
                    chosen = baseline.decide(
                        i, env.available_jobs, env.current_step, env.n_jobs
                    )
                    # Clip to valid range just in case
                    actions[agent_id] = min(int(chosen), env.n_jobs)

            obs, reward_dict, terms, truncs, _ = env.step(actions)
            step_reward += reward_dict['machine_0']
            step        += 1

            if terms['__all__'] or truncs['__all__']:
                break

        state = env.render()
        makespans.append(step)
        tardinesses.append(state['metrics']['tardiness_rate'])
        energies.append(state['metrics']['episode_energy'])
        completions.append(state['metrics']['completed'])
        rewards.append(step_reward)

    return {
        'mean_makespan':    float(np.mean(makespans)),
        'std_makespan':     float(np.std(makespans)),
        'mean_tardiness':   float(np.mean(tardinesses)),
        'std_tardiness':    float(np.std(tardinesses)),
        'mean_energy':      float(np.mean(energies)),
        'std_energy':       float(np.std(energies)),
        'mean_completions': float(np.mean(completions)),
        'mean_reward':      float(np.mean(rewards)),
    }


def run_mappo(model_path: str, env: FactoryGym, n_episodes: int) -> dict:
    """Run trained MAPPO model and collect metrics for comparison."""
    try:
        from agents.mappo_agent import MAPPOAgent
        agent = MAPPOAgent()
        agent.load(model_path)
        agent.set_eval_mode()
    except Exception as e:
        print(f"  [!] Could not load MAPPO model from {model_path}: {e}")
        return None

    makespans   = []
    tardinesses = []
    energies    = []
    completions = []
    rewards     = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=9000 + ep)
        step_reward = 0.0
        step = 0

        while True:
            masks   = {f'machine_{i}': env.get_action_mask(i)
                       for i in range(env.n_machines)}
            actions, _, _ = agent.predict(obs, masks)
            obs, rews, terms, truncs, _ = env.step(actions)
            step_reward += rews['machine_0']
            step        += 1

            if terms['__all__'] or truncs['__all__']:
                break

        state = env.render()
        makespans.append(step)
        tardinesses.append(state['metrics']['tardiness_rate'])
        energies.append(state['metrics']['episode_energy'])
        completions.append(state['metrics']['completed'])
        rewards.append(step_reward)

    return {
        'mean_makespan':  float(np.mean(makespans)),
        'std_makespan':   float(np.std(makespans)),
        'mean_tardiness': float(np.mean(tardinesses)),
        'std_tardiness':  float(np.std(tardinesses)),
        'mean_energy':    float(np.mean(energies)),
        'std_energy':     float(np.std(energies)),
        'mean_reward':    float(np.mean(rewards)),
    }


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs('logs', exist_ok=True)
    env     = FactoryGym()
    results = {}

    header = (
        f"\n{'='*72}\n"
        f"  Baseline Comparison  —  {args.episodes} episodes each\n"
        f"{'='*72}\n"
        f"{'Method':<8} {'Makespan':>12} {'Tardiness':>12} {'Energy':>12} {'Reward':>10}\n"
        f"{'-'*72}"
    )
    print(header)

    # Run classical baselines
    for method_name in args.methods:
        if method_name not in BASELINES:
            print(f"  Unknown method: {method_name}")
            continue

        print(f"  Running {method_name}...", end=' ', flush=True)
        metrics = run_one_baseline(
            BASELINES[method_name], env,
            n_episodes=args.episodes
        )
        results[method_name] = metrics
        print(
            f"\r{method_name:<8} "
            f"{metrics['mean_makespan']:>7.1f}±{metrics['std_makespan']:>4.1f}  "
            f"{metrics['mean_tardiness']:>8.1%}±{metrics['std_tardiness']:.1%}  "
            f"{metrics['mean_energy']:>7.2f}±{metrics['std_energy']:.2f}  "
            f"{metrics['mean_reward']:>9.2f}"
        )

    # Try MAPPO
    model_candidates = [
        os.path.join(args.model_path, 'mappo_factory_final.pth'),
        os.path.join(args.model_path, 'mappo_factory_best.pth'),
    ]
    for mp in model_candidates:
        if os.path.exists(mp):
            print(f"\n  Running MAPPO (from {mp})...", end=' ', flush=True)
            m = run_mappo(mp, env, n_episodes=args.episodes)
            if m:
                results['MAPPO'] = m
                print(
                    f"\r{'MAPPO':<8} "
                    f"{m['mean_makespan']:>7.1f}±{m['std_makespan']:>4.1f}  "
                    f"{m['mean_tardiness']:>8.1%}±{m['std_tardiness']:.1%}  "
                    f"{m['mean_energy']:>7.2f}±{m['std_energy']:.2f}  "
                    f"{m['mean_reward']:>9.2f}"
                )
            break

    print(f"\n{'='*72}")

    # Comparison analysis
    if 'MAPPO' in results and 'FCFS' in results:
        makespan_imp = (
            (results['FCFS']['mean_makespan'] - results['MAPPO']['mean_makespan'])
            / results['FCFS']['mean_makespan'] * 100
        )
        tardiness_imp = (
            results['FCFS']['mean_tardiness'] - results['MAPPO']['mean_tardiness']
        ) * 100
        energy_imp = (
            (results['FCFS']['mean_energy'] - results['MAPPO']['mean_energy'])
            / max(results['FCFS']['mean_energy'], 1e-9) * 100
        )
        print(f"\n  MAPPO vs FCFS improvements:")
        print(f"    Makespan:  {makespan_imp:+.1f}%")
        print(f"    Tardiness: {tardiness_imp:+.1f} pp")
        print(f"    Energy:    {energy_imp:+.1f}%")

        if 'EDD' in results:
            edd_tardiness_diff = (
                results['EDD']['mean_tardiness'] - results['MAPPO']['mean_tardiness']
            ) * 100
            print(f"    vs EDD Tardiness: {edd_tardiness_diff:+.1f} pp")

    # Save to CSV
    csv_path = os.path.join('logs', 'benchmark_results.csv')
    fields   = ['method', 'mean_makespan', 'std_makespan',
                 'mean_tardiness', 'std_tardiness',
                 'mean_energy', 'std_energy', 'mean_reward']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, m in results.items():
            row = {'method': method}
            row.update({k: round(v, 4) for k, v in m.items() if k in fields})
            w.writerow(row)

    print(f"\n  Results saved → {csv_path}\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',   type=int,  default=100)
    parser.add_argument('--methods',    nargs='+', default=['FCFS', 'SPT', 'EDD'])
    parser.add_argument('--model-path', type=str,  default='models/')
    args = parser.parse_args()
    main(args)