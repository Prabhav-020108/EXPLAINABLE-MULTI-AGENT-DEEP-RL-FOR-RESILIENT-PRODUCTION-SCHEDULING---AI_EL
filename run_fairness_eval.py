"""
run_fairness_eval.py
--------------------
Phase 4 evaluation runner.

Loads the best trained MAPPO model, runs N evaluation episodes,
records every job completion via FairnessAuditor, and produces:

    logs/fairness_audit.csv     — full job-level log
    logs/fairness_report.json   — metric summary with FAIR / BIAS_DETECTED

Usage
─────
    python run_fairness_eval.py                                       # defaults
    python run_fairness_eval.py --model models/mappo_factory_final.pth
    python run_fairness_eval.py --episodes 100 --seed 7000
    python run_fairness_eval.py --episodes 10  --seed 0              # quick test
"""

import argparse
import os
import sys
import numpy as np

# Allow running from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.factory_gym import FactoryGym
from agents.mappo_agent import MAPPOAgent
from fairness.auditor import FairnessAuditor
from fairness.reporter import FairnessReporter


# ════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════

def run_fairness_evaluation(
    model_path:  str = 'models/mappo_factory_best.pth',
    n_episodes:  int = 50,
    seed_offset: int = 5000,
) -> dict:
    """
    Run n_episodes with the trained MAPPO model and record fairness data.

    Args:
        model_path:  Path to the trained .pth checkpoint.
        n_episodes:  Number of episodes to evaluate (blueprint requires >= 50).
        seed_offset: Starting random seed (use a range not seen during training).

    Returns:
        The fairness report dictionary.
    """
    os.makedirs('logs', exist_ok=True)

    # Use training-level disruption rates so the model operates in the
    # conditions it was trained for. rush_rate=0.01 ensures ~15 rush orders
    # across 50 episodes, giving the Priority Fairness metric meaningful data.
    env_config = {
        'n_machines':        3,
        'n_jobs':            6,
        'max_steps':         100,
        'breakdown_rate':    0.02,
        'rush_rate':         0.01,
        'energy_spike_rate': 0.05,
    }

    # ── Banner ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Phase 4 — Fairness Evaluation")
    print(f"{'='*62}")
    print(f"  Model    : {model_path}")
    print(f"  Episodes : {n_episodes}")
    print(f"  Seed base: {seed_offset}")
    print(f"  Config   : breakdown={env_config['breakdown_rate']} "
          f"rush={env_config['rush_rate']} "
          f"spike={env_config['energy_spike_rate']}")

    # ── Load model and env ────────────────────────────────────────
    env   = FactoryGym(env_config)
    agent = MAPPOAgent()
    agent.load(model_path)
    agent.set_eval_mode()

    auditor  = FairnessAuditor()
    reporter = FairnessReporter(auditor, max_steps=env_config['max_steps'])

    ep_tardinesses = []
    ep_makespans   = []
    ep_energies    = []

    print(f"\n  Running episodes...\n")

    # ════════════════════════════════════════════════════════════════
    #  EPISODE LOOP
    # ════════════════════════════════════════════════════════════════
    for ep in range(n_episodes):

        obs, _ = env.reset(seed=seed_offset + ep)
        auditor.new_episode()  # tag all records with this episode number

        step = 0

        while True:
            # Build action masks for all agents
            masks = {
                f'machine_{i}': env.get_action_mask(i)
                for i in range(env.n_machines)
            }

            # Snapshot BEFORE step — to detect new completions
            count_before = len(env.completed_jobs)

            # Agent prediction and environment step
            actions, _, _ = agent.predict(obs, masks)
            obs, rewards, terms, truncs, _ = env.step(actions)

            # Record every job that completed during this single step
            for job in env.completed_jobs[count_before:]:
                if job.completion_time is not None:
                    auditor.record_job_completion(
                        job,
                        machine_id=(job.assigned_machine
                                    if job.assigned_machine is not None else 0),
                    )

            step += 1
            if terms['__all__'] or truncs['__all__']:
                break

        # Episode-level metrics for progress display
        state = env.render()
        ep_tardinesses.append(state['metrics']['tardiness_rate'])
        ep_makespans.append(step)
        ep_energies.append(state['metrics']['episode_energy'])

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(
                f"  Ep {ep+1:>3}/{n_episodes}  |  "
                f"Tardiness {state['metrics']['tardiness_rate']:.1%}  |  "
                f"Makespan {step:>3}  |  "
                f"Jobs logged {auditor.total_jobs_logged():>4}"
            )

    # ════════════════════════════════════════════════════════════════
    #  SAVE OUTPUTS
    # ════════════════════════════════════════════════════════════════
    print(f"\n  ── Evaluation summary {'─'*37}")
    print(f"  Mean tardiness : {np.mean(ep_tardinesses):.1%}")
    print(f"  Mean makespan  : {np.mean(ep_makespans):.1f} steps")
    print(f"  Mean energy    : {np.mean(ep_energies):.2f}")
    print(f"  Total jobs     : {auditor.total_jobs_logged()}")

    # 1. Save job-level CSV
    auditor.save_to_csv('logs/fairness_audit.csv')

    # 2. Generate and save JSON report (also prints terminal summary)
    report = reporter.save_report('logs/fairness_report.json')

    return report


# ════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 4: Run fairness evaluation on a trained MAPPO model'
    )
    parser.add_argument(
        '--model',
        default='models/mappo_factory_best.pth',
        help='Path to trained model checkpoint (.pth)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes (blueprint requires >= 50)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=5000,
        help='Random seed offset (must differ from training seeds 0-8999)',
    )
    args = parser.parse_args()

    run_fairness_evaluation(
        model_path  = args.model,
        n_episodes  = args.episodes,
        seed_offset = args.seed,
    )