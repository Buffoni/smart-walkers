"""
Train Alice (smart walker) against Bob (dumb walker) for each of the three
reward functions in parallel (one job per reward), then save results.

Parallelization strategy:
  - Games within one reward run are sequential (Alice must learn step-by-step).
  - The three reward-function runs are launched in parallel via joblib.
"""

import os
import yaml
import numpy as np
from joblib import Parallel, delayed

import swalkers

# --- Configuration ---
N_GAMES = int(1e4)
WORLD_DIMENSION = 11
ALICE_START = 1
BOB_START = WORLD_DIMENSION - 2
OUTPUT_DIR = "smart_alice"
ANNEALING = True
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 1.0
SOFTMAX_TEMPERATURE = 1.0
# ---------------------

REWARD_FUNCTIONS = [
    swalkers.rewards.linear_reward,
    swalkers.rewards.time_dependent_linear_reward,
    swalkers.rewards.sinusoidal_reward,
]


def play_with_smart_alice(reward_function):
    """Run N_GAMES sequentially via swalkers.simulation.playgames, training Alice."""
    reward_name = reward_function.__name__

    alice = swalkers.Walker(
        name="Alice",
        start_position=ALICE_START,
        world_dimension=WORLD_DIMENSION,
        use_brain=True,
        learn=True,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        softmax_temperature=SOFTMAX_TEMPERATURE,
    )
    bob = swalkers.Walker(
        name="Bob",
        start_position=BOB_START,
        world_dimension=WORLD_DIMENSION,
        use_brain=False,
        learn=False,
    )

    results = swalkers.simulation.playgames(
        walker_1=alice,
        walker_2=bob,
        world_dimension=WORLD_DIMENSION,
        reward_function=reward_function,
        number_of_games=N_GAMES,
        annealing=ANNEALING,
    )

    # Save results
    out_dir = os.path.join(OUTPUT_DIR, reward_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "meeting_squares.npy"), np.array(results["meeting_squares"]))
    np.save(os.path.join(out_dir, "alice_rewards.npy"), np.array(results["walker_1_rewards"]))
    np.save(os.path.join(out_dir, "bob_rewards.npy"), np.array(results["walker_2_rewards"]))
    np.save(os.path.join(out_dir, "policy_entropy.npy"), np.array(results["policy_entropy_list"]))
    np.save(os.path.join(out_dir, "A_entropy.npy"), np.array(results["A_entropy_list"]))
    np.save(os.path.join(out_dir, "alice_policy.npy"), alice.get_policy_tensor())
    np.save(os.path.join(out_dir, "bob_policy.npy"), bob.get_policy_tensor())
    np.save(os.path.join(out_dir, "alice_qtable.npy"), alice.q_table)
    np.save(os.path.join(out_dir, "bob_qtable.npy"), bob.q_table)

    # Post-training evaluation: freeze Alice's final policy (no learning, no annealing)
    # so that experimental meeting squares correspond to the saved policy tensor
    alice.learn = False
    alice.greedy = False
    bob_eval = swalkers.Walker(
        name="Bob",
        start_position=BOB_START,
        world_dimension=WORLD_DIMENSION,
        use_brain=False,
        learn=False,
    )
    eval_results = swalkers.simulation.playgames(
        walker_1=alice,
        walker_2=bob_eval,
        world_dimension=WORLD_DIMENSION,
        reward_function=reward_function,
        number_of_games=N_GAMES,
    )
    np.save(os.path.join(out_dir, "eval_meeting_squares.npy"), np.array(eval_results["meeting_squares"]))

    n_comp = results["number_of_compenetrations_prevented"]
    print(f"[{reward_name}] Compenetrations prevented: {n_comp} ({n_comp / N_GAMES * 100:.2f}%)")
    print(f"[{reward_name}] Cumulative Alice reward: {np.sum(results['walker_1_rewards']):.2f}")

    return reward_name


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    settings = {
        "n_games": N_GAMES,
        "world_dimension": WORLD_DIMENSION,
        "alice_start": ALICE_START,
        "bob_start": BOB_START,
        "annealing": ANNEALING,
        "learning_rate": LEARNING_RATE,
        "discount_factor": DISCOUNT_FACTOR,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "reward_functions": [f.__name__ for f in REWARD_FUNCTIONS],
    }
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    # Parallelize across the 3 reward functions
    Parallel(n_jobs=3)(
        delayed(play_with_smart_alice)(rf) for rf in REWARD_FUNCTIONS
    )

    print(f"\nAll results saved to '{OUTPUT_DIR}/'")
