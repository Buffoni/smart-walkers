"""
Simulate predator-prey games where both walkers are smart and trainable.
Alice (predator) wants to catch Bob (prey) quickly; Bob wants to delay the encounter.
The reward is time-dependent: Alice gets exp(-decay_rate * time), Bob gets the negative.
"""

import os
import yaml
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import swalkers

# --- Configuration ---
N_GAMES = int(1e4)
WORLD_DIMENSION = 11
ALICE_START = 1
BOB_START = WORLD_DIMENSION - 2
OUTPUT_DIR = "predator_prey_data"
N_JOBS = 16
SEED = 1000
ANNEALING = True
LEARNING_RATE = 0.5
# ---------------------

from functools import partial
REWARD_FUNCTION = partial(swalkers.rewards.predator_prey_reward)


def train_predator_prey(seed):
    """Train both walkers via Q-learning over N_GAMES sequential games."""
    np.random.seed(seed)

    alice = swalkers.Walker(
        name="Alice",
        start_position=ALICE_START,
        world_dimension=WORLD_DIMENSION,
        use_brain=True,
        learn=True,
        learning_rate=LEARNING_RATE
    )
    bob = swalkers.Walker(
        name="Bob",
        start_position=BOB_START,
        world_dimension=WORLD_DIMENSION,
        use_brain=True,
        learn=True,
        learning_rate=LEARNING_RATE
    )

    results = swalkers.simulation.playgames(
        walker_1=alice,
        walker_2=bob,
        world_dimension=WORLD_DIMENSION,
        reward_function=REWARD_FUNCTION,
        number_of_games=N_GAMES,
        annealing=ANNEALING,
        desc="Training predator-prey",
    )

    return results, alice, bob


def play_one_game_seeded(alice, bob, world_dimension, reward_function, seed):
    """Wrapper that seeds the worker RNG for reproducibility in parallel evaluation."""
    np.random.seed(seed)
    return swalkers.simulation.play_one_game(alice, bob, world_dimension, reward_function)


def evaluate_fixed_policies(alice, bob, task_seeds):
    """Evaluate N_GAMES in parallel with both policies frozen."""
    alice.learn = False
    bob.learn = False

    results = Parallel(n_jobs=N_JOBS)(
        delayed(play_one_game_seeded)(alice, bob, WORLD_DIMENSION, REWARD_FUNCTION, seed)
        for seed in tqdm(task_seeds, desc="Evaluating fixed policies")
    )

    meeting_squares = [r['meeting_square'] for r in results]
    alice_rewards = [r['walker_1_reward'] for r in results]
    bob_rewards = [r['walker_2_reward'] for r in results]
    compenetrations = [r['compenetration'] for r in results]

    n_comp = sum(compenetrations)
    print(f"Compenetrations: {n_comp} ({n_comp / len(results) * 100:.2f}%)")
    print(f"Cumulative Alice reward: {np.sum(alice_rewards):.2f}")
    print(f"Mean meeting time proxy (Alice reward): {np.mean(alice_rewards):.4f}")

    return meeting_squares, alice_rewards, bob_rewards


if __name__ == "__main__":
    rng = np.random.RandomState(SEED)
    training_seed = rng.randint(0, 2**31)
    eval_seeds = rng.randint(0, 2**31, size=N_GAMES)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = {
        "n_games": N_GAMES,
        "world_dimension": WORLD_DIMENSION,
        "alice_start": ALICE_START,
        "bob_start": BOB_START,
        "annealing": ANNEALING,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "reward_function": REWARD_FUNCTION.func.__name__,
    }
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    # --- Training ---
    results, alice, bob = train_predator_prey(training_seed)

    # Save training results
    np.save(os.path.join(OUTPUT_DIR, "meeting_squares.npy"), np.array(results["meeting_squares"]))
    np.save(os.path.join(OUTPUT_DIR, "alice_rewards.npy"), np.array(results["walker_1_rewards"]))
    np.save(os.path.join(OUTPUT_DIR, "bob_rewards.npy"), np.array(results["walker_2_rewards"]))
    np.save(os.path.join(OUTPUT_DIR, "policy_entropy.npy"), np.array(results["policy_entropy_list"]))
    np.save(os.path.join(OUTPUT_DIR, "bob_policy_entropy.npy"), np.array(results["walker_2_policy_entropy_list"]))
    np.save(os.path.join(OUTPUT_DIR, "A_entropy.npy"), np.array(results["A_entropy_list"]))
    np.save(os.path.join(OUTPUT_DIR, "alice_policy.npy"), alice.get_policy_tensor())
    np.save(os.path.join(OUTPUT_DIR, "bob_policy.npy"), bob.get_policy_tensor())
    np.save(os.path.join(OUTPUT_DIR, "alice_qtable.npy"), alice.q_table)
    np.save(os.path.join(OUTPUT_DIR, "bob_qtable.npy"), bob.q_table)

    # --- Evaluation with fixed policies ---
    meeting_squares, alice_rewards, bob_rewards = evaluate_fixed_policies(alice, bob, eval_seeds)

    np.save(os.path.join(OUTPUT_DIR, "fixed_meeting_squares.npy"), np.array(meeting_squares))
    np.save(os.path.join(OUTPUT_DIR, "fixed_alice_rewards.npy"), np.array(alice_rewards))
    np.save(os.path.join(OUTPUT_DIR, "fixed_bob_rewards.npy"), np.array(bob_rewards))

    print(f"\nAll results saved to '{OUTPUT_DIR}/'")
