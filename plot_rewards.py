import os
import numpy as np
import matplotlib.pyplot as plt

from swalkers.rewards import linear_reward, time_dependent_linear_reward, sinusoidal_reward

OUTPUT_DIR = "rewards_plots"
WORLD_DIMENSION = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)
squares = np.arange(WORLD_DIMENSION)

plt.rcParams.update({'font.size': 18}) # Set a larger font size for better readability
width = 0.8  # bar width, centered on each tick

# --- linear_reward ---
alice_vals = [linear_reward(s, WORLD_DIMENSION)[0] for s in squares]
bob_vals   = [linear_reward(s, WORLD_DIMENSION)[1] for s in squares]

fig, ax = plt.subplots()
ax.bar(squares, alice_vals, width=width, label="Alice", alpha=0.7)
ax.bar(squares, bob_vals,   width=width, label="Bob",   alpha=0.7)
ax.set_xlabel("Meeting square")
ax.set_ylabel("Reward")
ax.set_title("Linear reward")
ax.set_xticks(squares)
ax.legend()
ax.grid()
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUTPUT_DIR, "linear_reward.png"), bbox_inches="tight")
plt.close(fig)

# --- time_dependent_linear_reward ---
times = [0, 5, 10, 20]

fig, ax = plt.subplots()
for t in times:
    alice_vals_t = [time_dependent_linear_reward(s, WORLD_DIMENSION, time=t)[0] for s in squares]
    ax.bar(squares, alice_vals_t, width=width, label=f"t={t}", alpha=0.6)
ax.set_xlabel("Meeting square")
ax.set_ylabel("Alice reward")
ax.set_title("Time-dependent linear reward (Alice)")
ax.set_xticks(squares)
ax.legend()
ax.grid()
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUTPUT_DIR, "time_dependent_linear_reward.png"), bbox_inches="tight")
plt.close(fig)

# --- sinusoidal_reward ---
alice_vals = [sinusoidal_reward(s, WORLD_DIMENSION)[0] for s in squares]
bob_vals   = [sinusoidal_reward(s, WORLD_DIMENSION)[1] for s in squares]

fig, ax = plt.subplots()
ax.bar(squares, alice_vals, width=width, label="Alice", alpha=0.7)
ax.bar(squares, bob_vals,   width=width, label="Bob",   alpha=0.7)
ax.set_xlabel("Meeting square")
ax.set_ylabel("Reward")
ax.set_title("Sinusoidal reward")
ax.set_xticks(squares)
ax.legend()
ax.grid()
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUTPUT_DIR, "sinusoidal_reward.png"), bbox_inches="tight")
plt.close(fig)
