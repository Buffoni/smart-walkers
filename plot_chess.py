import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# -- I/O ------------------------------------------------------------------
DATA_DIR = "chess_data"
SAVE_DIR = "chess_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 20})

# -- Load settings ---------------------------------------------------------
with open(os.path.join(DATA_DIR, "settings.yaml")) as f:
    settings = yaml.safe_load(f)
total_moves = settings["total_moves"]

# -- Load data -------------------------------------------------------------
skill_levels  = np.load(os.path.join(DATA_DIR, "skill_levels.npy"))
entropy_means = np.load(os.path.join(DATA_DIR, "entropy_means.npy"))
entropy_stds  = np.load(os.path.join(DATA_DIR, "entropy_stds.npy"))

# -- Plot: Entropy vs Skill Level -----------------------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(skill_levels, entropy_means, yerr=entropy_stds,
             fmt="o", capsize=5, label="Mean Entropy")
plt.axhline(y=max(entropy_means), color="r", linestyle="--",
            label="Maximum Mean Entropy")
plt.axhline(y=min(entropy_means), color="g", linestyle="--",
            label="Minimum Mean Entropy")
plt.legend()
plt.xlabel("Stockfish Skill Level")
plt.ylabel(f"Entropy ({total_moves} moves)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "entropy_vs_skill.png"), bbox_inches="tight")
plt.close()

print(f"All plots saved to '{SAVE_DIR}/'")
