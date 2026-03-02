import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import swalkers

# -- I/O ------------------------------------------------------------------
DATA_DIR = "predator_prey_data"
SAVE_DIR = "predator_prey_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 18})

# -- Load settings ---------------------------------------------------------
with open(os.path.join(DATA_DIR, "settings.yaml")) as f:
    settings = yaml.safe_load(f)
world_dimension = settings["world_dimension"]
alice_start     = settings["alice_start"]
bob_start       = settings["bob_start"]

# -- Load data -------------------------------------------------------------
alice_policy_entropy = np.load(os.path.join(DATA_DIR, "policy_entropy.npy"))
bob_policy_entropy   = np.load(os.path.join(DATA_DIR, "bob_policy_entropy.npy"))
A_entropy            = np.load(os.path.join(DATA_DIR, "A_entropy.npy"))
alice_policy         = np.load(os.path.join(DATA_DIR, "alice_policy.npy"))
bob_policy           = np.load(os.path.join(DATA_DIR, "bob_policy.npy"))
fixed_meeting_squares = np.load(os.path.join(DATA_DIR, "fixed_meeting_squares.npy"))

# -- 1. Alice (predator) policy entropy over games -------------------------
plt.figure(figsize=(8, 6))
plt.plot(alice_policy_entropy)
plt.xlabel("Game Number")
plt.ylabel("Policy Entropy")
plt.title("Alice (Predator) Policy Entropy")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "alice_policy_entropy.png"), bbox_inches="tight")
plt.close()

# -- 2. Bob (prey) policy entropy over games -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(bob_policy_entropy)
plt.xlabel("Game Number")
plt.ylabel("Policy Entropy")
plt.title("Bob (Prey) Policy Entropy")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bob_policy_entropy.png"), bbox_inches="tight")
plt.close()

# -- 3. A-matrix entropy over games ---------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(A_entropy)
plt.xlabel("Game Number")
plt.ylabel("Configuration Entropy")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "A_entropy.png"), bbox_inches="tight")
plt.close()

# -- 4. First encounter position: theoretical vs experimental (fixed) ------
P0 = swalkers.functions.calculate_starting_distribution_in_tensorspace(
    walker_1_start_pos=alice_start,
    walker_2_start_pos=bob_start,
    world_dimension=world_dimension,
)

p = swalkers.functions.calculate_first_encounter_probabilities(
    starting_distribution=P0,
    walker_1_policy_tensor=alice_policy,
    walker_2_policy_tensor=bob_policy,
)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(world_dimension), p, marker="o", linestyle="-",
         color="red", label="Theoretical")
plt.hist(fixed_meeting_squares,
         bins=np.arange(world_dimension + 1) - 0.5,
         density=True, alpha=0.7, color="blue", edgecolor="black",
         label="Experimental (fixed policies)")
plt.xlabel("Site Index")
plt.ylabel("Probability")
plt.xticks(np.arange(world_dimension))
plt.xlim(-0.5, world_dimension - 0.5)
plt.title("Predator-Prey: First Encounter Position")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "first_encounter_position.png"), bbox_inches="tight")
plt.close()

# -- 5. Time matrix -------------------------------------------------------
alice_pt = np.copy(alice_policy)
bob_pt   = np.copy(bob_policy)
swalkers.utilities.impose_reflective_boundary_conditions(alice_pt)
swalkers.utilities.impose_reflective_boundary_conditions(bob_pt)
A = swalkers.utilities.build_matrix_A(alice_pt, bob_pt)
A = swalkers.utilities.remove_forbidden_compenetration_processes(A)
time_vector, time_matrix = swalkers.functions.calculate_first_encounter_times(A)

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
ax.set_title("Predator-Prey: Expected Meeting Time")
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "time_matrix.png"),
            bbox_inches="tight", pad_inches=0.5)
plt.close()

print(f"All plots saved to '{SAVE_DIR}/'")
