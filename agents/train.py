"""
train.py
--------
MAPPO Training Script — runs on Google Colab GPU or local CPU.

Quick local smoke test (verify pipeline works, ~1 min):
    python agents/train.py --steps 5000 --log-interval 1000 --eval-episodes 3

Full training on Google Colab (1M steps, ~2-4 hours):
    python agents/train.py --steps 1000000 --log-interval 50000 --save-path /content/drive/MyDrive/factory_rl_models/
"""

import argparse
import os
import sys
import csv
import time
import numpy as np
import torch

# Allow imports from project root regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.factory_gym import FactoryGym
from agents.mappo_agent import MAPPOAgent


# ────────────────────────────────────────────────────────────────────
#  EVALUATION HELPER
# ────────────────────────────────────────────────────────────────────

def evaluate(env: FactoryGym, agent: MAPPOAgent, n_episodes: int = 10) -> dict:
    """
    Run n_episodes with the current policy (no exploration) and return metrics.
    Uses seeds 9000+ so eval episodes are never seen during training.
    """
    agent.set_eval_mode()

    all_rewards   = []
    all_tardiness = []
    all_energy    = []
    all_makespans = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=9000 + ep)
        total_reward = 0.0
        steps        = 0

        while True:
            masks = {
                f'machine_{i}': env.get_action_mask(i)
                for i in range(env.n_machines)
            }
            actions, _, _ = agent.predict(obs, masks)
            obs, rewards, terms, truncs, _ = env.step(actions)
            total_reward += rewards['machine_0']
            steps        += 1

            if terms['__all__'] or truncs['__all__']:
                break

        state = env.render()
        all_rewards.append(total_reward)
        all_tardiness.append(state['metrics']['tardiness_rate'])
        all_energy.append(state['metrics']['episode_energy'])
        all_makespans.append(steps)

    agent.set_train_mode()

    return {
        'mean_reward':    float(np.mean(all_rewards)),
        'std_reward':     float(np.std(all_rewards)),
        'mean_tardiness': float(np.mean(all_tardiness)),
        'mean_energy':    float(np.mean(all_energy)),
        'mean_makespan':  float(np.mean(all_makespans)),
    }


# ────────────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup directories ─────────────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Config ───────────────────────────────────────────────────
    env_config = {
        'n_machines':        args.n_machines,
        'n_jobs':            args.n_jobs,
        'max_steps':         args.max_steps,
        'breakdown_rate':    0.02,
        'rush_rate':         0.01,
        'energy_spike_rate': 0.05,
    }

    agent_config = {
        'learning_rate': args.lr,
        'gamma':         args.gamma,
        'clip_ratio':    args.clip,
        'entropy_coef':  args.entropy,
        'batch_size':    args.batch_size,
        'n_epochs':      args.n_epochs,
        'max_grad_norm': 0.5,
    }

    # ── Environments ─────────────────────────────────────────────
    train_env = FactoryGym(env_config)
    eval_env  = FactoryGym(env_config)

    # ── Agent ────────────────────────────────────────────────────
    agent = MAPPOAgent(
        obs_dim   = 38,
        n_actions = train_env.n_jobs + 1,
        n_agents  = train_env.n_machines,
        config    = agent_config,
    )

    # ── Logging setup ─────────────────────────────────────────────
    log_path = os.path.join('logs', 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'step', 'mean_reward', 'std_reward',
            'mean_tardiness', 'mean_energy', 'mean_makespan',
            'actor_loss', 'critic_loss', 'entropy', 'elapsed_min'
        ])

    # ── Training state ────────────────────────────────────────────
    best_reward  = -float('inf')
    total_steps  = 0
    n_updates    = 0
    start_time   = time.time()

    print(f"\n{'='*60}")
    print(f"  MAPPO Factory Scheduling — Training")
    print(f"{'='*60}")
    print(f"  Device       : {agent.device}")
    print(f"  Target steps : {args.steps:,}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Save path    : {args.save_path}")
    print(f"{'='*60}\n")

    # ── Initial evaluation (random policy baseline) ───────────────
    print("Running initial evaluation (random policy)...")
    init_metrics = evaluate(eval_env, agent, n_episodes=args.eval_episodes)
    print(f"  Initial reward: {init_metrics['mean_reward']:.2f} | "
          f"Tardiness: {init_metrics['mean_tardiness']:.1%}\n")

    # ── Reset environment for training ────────────────────────────
    obs, _ = train_env.reset(seed=42)
    last_losses  = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
    last_log_step = 0   # track when we last logged

    # ════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ════════════════════════════════════════════════════════════
    while total_steps < args.steps:

        # ── Collect rollout ───────────────────────────────────────
        agent.buffer.clear()
        agent.set_train_mode()

        for _ in range(args.rollout_steps):
            masks = {
                f'machine_{i}': train_env.get_action_mask(i)
                for i in range(train_env.n_machines)
            }

            actions, log_probs, global_state, value = agent.predict_with_value(
                obs, masks
            )

            next_obs, rewards, terms, truncs, _ = train_env.step(actions)

            done = terms['__all__'] or truncs['__all__']
            agent.buffer.add(
                obs_dict      = obs,
                actions_dict  = actions,
                log_probs_dict= log_probs,
                rewards_dict  = rewards,
                done          = done,
                global_state  = global_state,
                value         = value,
                masks_dict    = masks,
            )

            total_steps += 1
            obs          = next_obs

            if done:
                obs, _ = train_env.reset()

            if total_steps >= args.steps:
                break

        # ── Bootstrap value for last observation ──────────────────
        with torch.no_grad():
            gs = np.concatenate([
                obs[f'machine_{i}'] for i in range(train_env.n_machines)
            ])
            gs_t      = torch.FloatTensor(gs).unsqueeze(0).to(agent.device)
            next_val  = float(agent.critic(gs_t).item())

        # ── PPO update ────────────────────────────────────────────
        last_losses = agent.update(next_val)
        n_updates  += 1

        # ── Log and evaluate ──────────────────────────────────────
        if (total_steps - last_log_step) >= args.log_interval or total_steps >= args.steps:
            metrics  = evaluate(eval_env, agent, n_episodes=args.eval_episodes)
            elapsed  = (time.time() - start_time) / 60.0

            print(
                f"Step {total_steps:>8,}  |  "
                f"Reward {metrics['mean_reward']:>8.2f}±{metrics['std_reward']:.1f}  |  "
                f"Tardy {metrics['mean_tardiness']:>5.1%}  |  "
                f"Makespan {metrics['mean_makespan']:>5.1f}  |  "
                f"Energy {metrics['mean_energy']:>5.2f}  |  "
                f"{elapsed:.1f}min"
            )

            # CSV logging
            with open(log_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    total_steps,
                    round(metrics['mean_reward'],    4),
                    round(metrics['std_reward'],     4),
                    round(metrics['mean_tardiness'], 4),
                    round(metrics['mean_energy'],    4),
                    round(metrics['mean_makespan'],  2),
                    round(last_losses['actor_loss'],  4),
                    round(last_losses['critic_loss'], 4),
                    round(last_losses['entropy'],     4),
                    round(elapsed, 2),
                ])

            # Save best model
            if metrics['mean_reward'] > best_reward:
                best_reward  = metrics['mean_reward']
                best_path    = os.path.join(args.save_path, 'mappo_factory_best.pth')
                agent.save(best_path)
                print(f"          → New best! Reward={best_reward:.2f} saved.")
            last_log_step = total_steps   # ← update after each log

    # ── Save final model ──────────────────────────────────────────
    final_path = os.path.join(args.save_path, 'mappo_factory_final.pth')
    agent.save(final_path)
    total_min = (time.time() - start_time) / 60.0
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Updates:     {n_updates}")
    print(f"  Total time:  {total_min:.1f} minutes")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Final model: {final_path}")
    print(f"{'='*60}\n")

    return agent


# ────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSER
# ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MAPPO agents for factory scheduling'
    )

    # Environment
    parser.add_argument('--n-machines',    type=int,   default=3)
    parser.add_argument('--n-jobs',        type=int,   default=6)
    parser.add_argument('--max-steps',     type=int,   default=100)

    # Training
    parser.add_argument('--steps',         type=int,   default=1_000_000,
                        help='Total environment steps')
    parser.add_argument('--rollout-steps', type=int,   default=2048,
                        help='Steps per rollout before PPO update')
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--n-epochs',      type=int,   default=10)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--clip',          type=float, default=0.2)
    parser.add_argument('--entropy',       type=float, default=0.01)

    # Logging
    parser.add_argument('--log-interval',  type=int,   default=50_000)
    parser.add_argument('--eval-episodes', type=int,   default=10)
    parser.add_argument('--save-path',     type=str,   default='models/')

    args = parser.parse_args()
    train(args)