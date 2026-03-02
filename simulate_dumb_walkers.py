"""
Simulate 100000 games between dumb (random) walkers in parallel and save
the first encounter time and position for each game to the dumb_walkers/ folder.
"""

import os
import yaml
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from swalkers import Walker
from swalkers.utilities import check_end_game

# --- Configuration ---
N_GAMES = int(1e5)
WORLD_DIMENSION = 11
WALKER_1_START = 1
WALKER_2_START = WORLD_DIMENSION - 2
OUTPUT_DIR = "dumb_walkers"
N_JOBS = 16
# ---------------------


def play_one_dumb_game(_):
    """Play a single game between two dumb (random) walkers. Returns (time, position)."""
    alice = Walker("Alice", WALKER_1_START, WORLD_DIMENSION, use_brain=False, learn=False)
    bob = Walker("Bob", WALKER_2_START, WORLD_DIMENSION, use_brain=False, learn=False)

    steps = 0
    while True:
        alice.move(bob.position)
        bob.move(alice.position)
        steps += 1

        end_game, _ = check_end_game(alice, bob)
        if end_game:
            return steps, alice.position, alice.get_policy_tensor(), bob.get_policy_tensor()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    settings = {
        "n_games": N_GAMES,
        "world_dimension": WORLD_DIMENSION,
        "walker_1_start": WALKER_1_START,
        "walker_2_start": WALKER_2_START
    }
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    results = Parallel(n_jobs=N_JOBS)(delayed(play_one_dumb_game)(_) for _ in tqdm(range(N_GAMES), desc="Simulating games"))

    times, positions, alice_policies, bob_policies = zip(*results)

    np.save(os.path.join(OUTPUT_DIR, "first_encounter_times.npy"), np.array(times))
    np.save(os.path.join(OUTPUT_DIR, "first_encounter_positions.npy"), np.array(positions))

    # Save just one copy of the policies (they are all the same since the walkers are dumb)
    np.save(os.path.join(OUTPUT_DIR, "alice_policy.npy"), np.array(alice_policies[0]))
    np.save(os.path.join(OUTPUT_DIR, "bob_policy.npy"), np.array(bob_policies[0]))

    print(f"Saved {N_GAMES} games to '{OUTPUT_DIR}/'")
    print(f"  Mean encounter time:     {np.mean(times):.2f}")
    print(f"  Mean encounter position: {np.mean(positions):.2f}")
