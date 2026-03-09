import os
import yaml
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from swalkers import Walker, play_one_game

# --- Configuration ---
N_GAMES = int(1e5)
WORLD_DIMENSION = 11
WALKER_1_START = 1
WALKER_2_START = WORLD_DIMENSION - 2
OUTPUT_DIR = "dumb_walkers_data"
N_JOBS = 16
SEED = 1000
# ---------------------

def play_one_dumb_game(seed):
    """Play a single game between two dumb (random) walkers."""
    np.random.seed(seed)
    alice = Walker("Alice", WALKER_1_START, WORLD_DIMENSION, use_brain=False, learn=False)
    bob = Walker("Bob", WALKER_2_START, WORLD_DIMENSION, use_brain=False, learn=False)

    dummy_reward = lambda meeting_square, world_dimension, time=None: (0, 0)
    result = play_one_game(alice, bob, WORLD_DIMENSION, dummy_reward)
    
    return result['moves'], result['meeting_square'], alice.get_policy_tensor(), bob.get_policy_tensor()

if __name__ == "__main__":
    # Generate deterministic per-task seeds from the master seed.
    # np.random.seed() only affects the parent process; joblib workers
    # spawn separate processes that don't inherit the parent's random state.
    rng = np.random.RandomState(SEED)
    task_seeds = rng.randint(0, 2**31, size=N_GAMES)

    settings = {
        "n_games": N_GAMES,
        "world_dimension": WORLD_DIMENSION,
        "walker_1_start": WALKER_1_START,
        "walker_2_start": WALKER_2_START,
        "seed": SEED,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    results = Parallel(n_jobs=N_JOBS)(delayed(play_one_dumb_game)(seed) for seed in tqdm(task_seeds, desc="Simulating games"))

    times, positions, alice_policies, bob_policies = zip(*results)

    np.save(os.path.join(OUTPUT_DIR, "first_encounter_times.npy"), np.array(times))
    np.save(os.path.join(OUTPUT_DIR, "first_encounter_positions.npy"), np.array(positions))
    np.save(os.path.join(OUTPUT_DIR, "alice_policy.npy"), np.array(alice_policies[0]))
    np.save(os.path.join(OUTPUT_DIR, "bob_policy.npy"), np.array(bob_policies[0]))

    print(f"Saved {N_GAMES} games to '{OUTPUT_DIR}/'")
