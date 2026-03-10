import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import swalkers

SAVE_DIR = "dumb_walkers_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

FONT_SIZE = 22
FIGSIZE = (9, 7)
DPI = 100
plt.rcParams.update({'font.size': FONT_SIZE, 'figure.dpi': DPI}) # Set a larger font size for better readability and fixed DPI

positions = np.load(os.path.join("dumb_walkers_data", "first_encounter_positions.npy"))
alice_policy_tensor = np.load(os.path.join("dumb_walkers_data", "alice_policy.npy"))
bob_policy_tensor = np.load(os.path.join("dumb_walkers_data", "bob_policy.npy"))
with open(os.path.join("dumb_walkers_data", "settings.yaml"), "r") as f:
    settings = yaml.safe_load(f)

world_dimension = settings["world_dimension"]
alice_start = settings["walker_1_start"]
bob_start = settings["walker_2_start"]

P0 = swalkers.functions.calculate_starting_distribution_in_tensorspace(
    walker_1_start_pos=alice_start,
    walker_2_start_pos=bob_start,
    world_dimension=world_dimension
)

p = swalkers.functions.calculate_first_encounter_probabilities(
    starting_distribution=P0,
    walker_1_policy_tensor=alice_policy_tensor,
    walker_2_policy_tensor=bob_policy_tensor,
)

#------------------------------------------------------------------
# FIRST ENCOUNTER POSITION DISTRIBUTION PLOT

plt.figure(figsize=(12, 6))
plt.plot(
    np.arange(world_dimension),
    p,
    marker='o',
    linestyle='-',
    color='red',
    label='Theoretical Probabilities'
    )
plt.hist(
    positions,
    bins=np.arange(-0.5, world_dimension + 0.5, 1),
    density=True,
    color='C0',
    edgecolor='black',
    label='Experimental Probabilities'
    )
plt.xlabel('Site Index')
plt.ylabel('Probability')
plt.xticks(np.arange(0, world_dimension, 1))
plt.xlim(-0.5, world_dimension - 1 + 0.5)
plt.grid()
plt.legend(fontsize=plt.rcParams['font.size'], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "first_encounter_position_distribution.png"), bbox_inches="tight")
plt.close()
#------------------------------------------------------------------

#------------------------------------------------------------------
# FIRST ENCOUNTER TIME ANALYSIS -> BAR PLOT OF THE TIME VECTOR

swalkers.utilities.impose_reflective_boundary_conditions(alice_policy_tensor)
swalkers.utilities.impose_reflective_boundary_conditions(bob_policy_tensor)
A = swalkers.utilities.build_matrix_A(alice_policy_tensor, bob_policy_tensor)
time_vector, time_matrix = swalkers.functions.calculate_first_encounter_times(A)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.bar(
    np.arange(world_dimension**2),
    time_vector,
    color='C0',
    label='Time to First Encounter'
    )
ax.set_xlabel('Meeting Site Index', fontsize=FONT_SIZE)
ax.set_ylabel('Time to First Encounter', fontsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=FONT_SIZE)
ax.legend(fontsize=FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=1)
ax.grid(axis='y', alpha=0.75)
fig.subplots_adjust(top=0.85)
fig.savefig(os.path.join(SAVE_DIR, "first_encounter_time_vector.png"), dpi=DPI)
plt.close(fig)
#------------------------------------------------------------------

#------------------------------------------------------------------
# FIRST ENCOUNTER TIME ANALYSIS -> 3D BAR PLOT OF THE TIME MATRIX

time_matrix = np.array(time_matrix)
nrows, ncols = time_matrix.shape

# Create a meshgrid for bar positions
xpos, ypos = np.meshgrid(np.arange(ncols), np.arange(nrows), indexing="xy")
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)  # All bars start at height 0

# Bar dimensions
dx = dy = 0.8  # Width and depth of bars
dz = time_matrix.flatten()  # Heights of the bars

# Plotting
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos-0.5, ypos-0.5, zpos, dx, dy, dz, shade=True, edgecolor='black', color='C0')
ax.set_xlabel('Alice Position', labelpad=15, fontsize=FONT_SIZE)
ax.set_ylabel('Bob Position', labelpad=15, fontsize=FONT_SIZE)
ax.zaxis.set_tick_params(labelsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=FONT_SIZE)
ax.invert_xaxis()  # Invert x-axis to match the original matrix orientation

# Create a legend for the 3D bars using a proxy Patch so it doesn't overlap the plot
proxy = Patch(facecolor='C0', edgecolor='black', label='Time to First Encounter')
ax.legend(handles=[proxy], fontsize=FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=1)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.88)
fig.savefig(os.path.join(SAVE_DIR, "first_encounter_time_matrix.png"), dpi=DPI)
plt.close(fig)
#------------------------------------------------------------------
