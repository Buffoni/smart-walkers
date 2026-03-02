import os
import numpy as np
import matplotlib.pyplot as plt

from swalkers.rewards import linear_reward, time_dependent_linear_reward, sinusoidal_reward, predator_prey_reward

OUTPUT_DIR = "rewards_plots"
WORLD_DIMENSION = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)
squares = np.arange(WORLD_DIMENSION)

plt.rcParams.update({'font.size': 18}) # Set a larger font size for better readability
width = 0.8  # bar width, centered on each tick

# --- linear_reward ---
alice_vals = [linear_reward(s, WORLD_DIMENSION)[0] for s in squares]
bob_vals   = [linear_reward(s, WORLD_DIMENSION)[1] for s in squares]

plt.figure()
plt.bar(squares, alice_vals, width=width, label="Alice", alpha=0.7)
plt.bar(squares, bob_vals,   width=width, label="Bob",   alpha=0.7)
plt.xlabel("Meeting square")
plt.ylabel("Reward")
plt.title("Linear reward")
plt.xticks(squares)
plt.legend()
plt.grid()
plt.gca().set_axisbelow(True)
plt.savefig(os.path.join(OUTPUT_DIR, "linear_reward.png"), bbox_inches="tight")
plt.close()

# --- time_dependent_linear_reward ---
times = [0, 5, 10, 20]

plt.figure()
for t in times:
    alice_vals_t = [time_dependent_linear_reward(s, WORLD_DIMENSION, time=t)[0] for s in squares]
    plt.bar(squares, alice_vals_t, width=width, label=f"t={t}", alpha=0.6)
plt.xlabel("Meeting square")
plt.ylabel("Alice reward")
plt.title("Time-dependent linear reward (Alice)")
plt.xticks(squares)
plt.legend()
plt.grid()
plt.gca().set_axisbelow(True)
plt.savefig(os.path.join(OUTPUT_DIR, "time_dependent_linear_reward.png"), bbox_inches="tight")
plt.close()

# --- sinusoidal_reward ---
alice_vals = [sinusoidal_reward(s, WORLD_DIMENSION)[0] for s in squares]
bob_vals   = [sinusoidal_reward(s, WORLD_DIMENSION)[1] for s in squares]

plt.figure()
plt.bar(squares, alice_vals, width=width, label="Alice", alpha=0.7)
plt.bar(squares, bob_vals,   width=width, label="Bob",   alpha=0.7)
plt.xlabel("Meeting square")
plt.ylabel("Reward")
plt.title("Sinusoidal reward")
plt.xticks(squares)
plt.legend()
plt.grid()
plt.gca().set_axisbelow(True)
plt.savefig(os.path.join(OUTPUT_DIR, "sinusoidal_reward.png"), bbox_inches="tight")
plt.close()

# --- predator_prey_reward ---
# This reward depends only on meeting time, not on meeting square.
# Plot Alice's reward as a function of time: exp(-decay_rate * t)
times = np.arange(0, 301)
alice_vals = [predator_prey_reward(0, WORLD_DIMENSION, time=t)[0] for t in times]
bob_vals   = [predator_prey_reward(0, WORLD_DIMENSION, time=t)[1] for t in times]

plt.figure()
plt.plot(times, alice_vals, label="Alice (predator)", linewidth=2)
plt.plot(times, bob_vals,   label="Bob (prey)",       linewidth=2)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Meeting time")
plt.ylabel("Reward")
plt.title("Predator-prey reward")
plt.legend()
plt.grid()
plt.gca().set_axisbelow(True)
plt.savefig(os.path.join(OUTPUT_DIR, "predator_prey_reward.png"), bbox_inches="tight")
plt.close()
