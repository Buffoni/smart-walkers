import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import swalkers

# ── I/O ─────────────────────────────────────────────────────────────────────
DATA_DIR = "smart_alice_data"
SAVE_DIR = "smart_alice_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 20})

# ── Load settings ────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "settings.yaml")) as f:
    settings = yaml.safe_load(f)
world_dimension = settings["world_dimension"]
alice_start     = settings["alice_start"]
bob_start       = settings["bob_start"]
reward_names    = settings["reward_functions"]

# ── Load per-reward data ─────────────────────────────────────────────────────
data = {}
for name in reward_names:
    d = os.path.join(DATA_DIR, name)
    data[name] = {
        "meeting_squares":  np.load(os.path.join(d, "meeting_squares.npy")),
        "alice_rewards":    np.load(os.path.join(d, "alice_rewards.npy")),
        "bob_rewards":      np.load(os.path.join(d, "bob_rewards.npy")),
        "policy_entropy":   np.load(os.path.join(d, "policy_entropy.npy")),
        "A_entropy":        np.load(os.path.join(d, "A_entropy.npy")),
        "alice_policy":     np.load(os.path.join(d, "alice_policy.npy")),
        "bob_policy":       np.load(os.path.join(d, "bob_policy.npy")),
        "fixed_meeting_squares": np.load(os.path.join(d, "fixed_meeting_squares.npy")),
    }

def label(name):
    return name.replace("_", " ").title()

# ── 1. Cumulative rewards ────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
for name, d in data.items():
    plt.plot(np.cumsum(d["alice_rewards"]), label=label(name))
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Game Number")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cumulative_rewards.png"), bbox_inches="tight")
plt.close()

# ── 2. Policy entropy over games ─────────────────────────────────────────────
plt.figure(figsize=(8, 8))
for name, d in data.items():
    plt.plot(d["policy_entropy"], label=label(name))
plt.xlabel("Game Number")
plt.ylabel("Policy Entropy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "policy_entropy.png"), bbox_inches="tight")
plt.close()

# ── 3. A-matrix entropy over games ───────────────────────────────────────────
plt.figure(figsize=(8, 8))
for name, d in data.items():
    plt.plot(d["A_entropy"], label=label(name))
plt.xlabel("Game Number")
plt.ylabel("Configuration Entropy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "A_entropy.png"), bbox_inches="tight")
plt.close()

# ── 4. First encounter position: theoretical vs training experimental ─────────
P0 = swalkers.functions.calculate_starting_distribution_in_tensorspace(
    walker_1_start_pos=alice_start,
    walker_2_start_pos=bob_start,
    world_dimension=world_dimension,
)

for name, d in data.items():
    p = swalkers.functions.calculate_first_encounter_probabilities(
        starting_distribution=P0,
        walker_1_policy_tensor=d["alice_policy"],
        walker_2_policy_tensor=d["bob_policy"],
    )

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(world_dimension), p, marker="o", linestyle="-",
             color="red", label="Theoretical")
    plt.hist(d["fixed_meeting_squares"],
             bins=np.arange(world_dimension + 1) - 0.5,
             density=True, alpha=0.7, color="blue", edgecolor="black",
             label="Experimental (final policy)")
    plt.xlabel("Site Index")
    plt.ylabel("Probability")
    plt.xticks(np.arange(world_dimension))
    plt.xlim(-0.5, world_dimension - 0.5)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"first_encounter_position_{name}.png"), bbox_inches="tight")
    plt.close()

# ── 5. Time vector and time matrix ───────────────────────────────────────
for name, d in data.items():
    alice_pt = np.copy(d["alice_policy"])
    bob_pt   = np.copy(d["bob_policy"])
    swalkers.utilities.impose_reflective_boundary_conditions(alice_pt)
    swalkers.utilities.impose_reflective_boundary_conditions(bob_pt)
    A = swalkers.utilities.build_matrix_A(alice_pt, bob_pt)
    time_vector, time_matrix = swalkers.functions.calculate_first_encounter_times(A)

    # 3D bar plot of time matrix
    matrix = np.array(time_matrix)
    nrows, ncols = matrix.shape
    xpos, ypos = np.meshgrid(np.arange(ncols), np.arange(nrows), indexing="xy")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8
    dz = matrix.flatten()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos - 0.5, ypos - 0.5, zpos, dx, dy, dz, shade=True)
    ax.set_xlabel("Alice Position", labelpad=15)
    ax.set_ylabel("Bob Position",   labelpad=15)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"time_matrix_{name}.png"),
                bbox_inches="tight", pad_inches=0.5)
    plt.close()

print(f"All plots saved to '{SAVE_DIR}/'")
