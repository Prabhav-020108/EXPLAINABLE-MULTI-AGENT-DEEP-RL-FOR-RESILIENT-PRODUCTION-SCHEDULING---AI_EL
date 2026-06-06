# Save as plot_training.py
import csv
import matplotlib.pyplot as plt
import numpy as np

steps      = []
rewards    = []
tardinesses= []
actor_loss = []
critic_loss= []

with open('logs/training_log.csv') as f:
    for row in csv.DictReader(f):
        steps.append(int(row['step']))
        rewards.append(float(row['mean_reward']))
        tardinesses.append(float(row['mean_tardiness']) * 100)
        actor_loss.append(abs(float(row['actor_loss'])))
        critic_loss.append(float(row['critic_loss']))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('MAPPO Training Progress — 1M Steps', fontsize=14, fontweight='bold')

# ── Reward ────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(steps, rewards, 'b-o', markersize=4, linewidth=1.5)
ax.axhline(y=max(rewards), color='green', linestyle='--',
           label=f'Best: {max(rewards):.2f}', alpha=0.7)
z = np.polyfit(steps, rewards, 1)
trend = np.poly1d(z)
ax.plot(steps, trend(steps), 'r--', alpha=0.5, label='Trend')
ax.set_title('Mean Reward')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Reward')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Tardiness ─────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(steps, tardinesses, 'orange', marker='s', markersize=4, linewidth=1.5)
ax.axhline(y=15, color='red', linestyle='--',
           label='Target: 15%', linewidth=2)
ax.axhline(y=min(tardinesses), color='green', linestyle='--',
           label=f'Best: {min(tardinesses):.1f}%', alpha=0.7)
ax.set_title('Tardiness Rate (%)')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Tardiness %')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(tardinesses) * 1.2)

# ── Actor Loss ────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(steps, actor_loss, 'purple', marker='^', markersize=4, linewidth=1.5)
ax.set_title('|Actor Loss| (increasing = learning)')
ax.set_xlabel('Training Steps')
ax.set_ylabel('|Actor Loss|')
ax.grid(True, alpha=0.3)

# ── Critic Loss ───────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(steps, critic_loss, 'teal', marker='D', markersize=4, linewidth=1.5)
ax.set_title('Critic Loss (decreasing = converging)')
ax.set_xlabel('Training Steps')
ax.set_ylabel('MSE Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/training_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training curve saved to logs/training_curve.png")