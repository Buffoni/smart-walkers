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
OUTPUT_DIR = "smart_alice_data"
N_JOBS = 8
SEED = 1000
ANNEALING = True
LEARNING_RATE = 0.5
# ---------------------

REWARD_FUNCTIONS = [
    swalkers.rewards.linear_reward,
    swalkers.rewards.time_dependent_linear_reward,
    swalkers.rewards.sinusoidal_reward,
]

def play_with_learning_alice(reward_function, seed):
    """Run N_GAMES sequentially via swalkers.simulation.playgames, training Alice."""
    np.random.seed(seed)
    reward_name = reward_function.__name__

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

    meeting_squares_path = os.path.join(out_dir, "meeting_squares.npy")
    alice_rewards_path = os.path.join(out_dir, "alice_rewards.npy")
    bob_rewards_path = os.path.join(out_dir, "bob_rewards.npy")
    policy_entropy_path = os.path.join(out_dir, "policy_entropy.npy")
    A_entropy_path = os.path.join(out_dir, "A_entropy.npy")
    alice_policy_path = os.path.join(out_dir, "alice_policy.npy")
    bob_policy_path = os.path.join(out_dir, "bob_policy.npy")
    alice_qtable_path = os.path.join(out_dir, "alice_qtable.npy")
    bob_qtable_path = os.path.join(out_dir, "bob_qtable.npy")

    np.save(meeting_squares_path, np.array(results["meeting_squares"]))
    np.save(alice_rewards_path, np.array(results["walker_1_rewards"]))
    np.save(bob_rewards_path, np.array(results["walker_2_rewards"]))
    np.save(policy_entropy_path, np.array(results["policy_entropy_list"]))
    np.save(A_entropy_path, np.array(results["A_entropy_list"]))
    np.save(alice_policy_path, alice.get_policy_tensor())
    np.save(bob_policy_path, bob.get_policy_tensor())
    np.save(alice_qtable_path, alice.q_table)
    np.save(bob_qtable_path, bob.q_table)

    return reward_name, alice, bob

def play_one_game_seeded(alice, bob, world_dimension, reward_function, seed):
    """Wrapper around play_one_game that seeds the worker's RNG for reproducibility."""
    np.random.seed(seed)
    return swalkers.simulation.play_one_game(alice, bob, world_dimension, reward_function)

def playing_with_smart_alice(reward_function, alice, bob, task_seeds):
    """Play N_GAMES in parallel with Alice's learned policy frozen and Bob as a dumb walker."""
    reward_name = reward_function.__name__
    alice.learn = False

    results = Parallel(n_jobs=N_JOBS)(
        delayed(play_one_game_seeded)(alice, bob, WORLD_DIMENSION, reward_function, seed) for seed in tqdm(task_seeds, desc=f"Fixed Alice [{reward_name}]")
    )

    meeting_squares = [r['meeting_square'] for r in results]
    alice_rewards = [r['walker_1_reward'] for r in results]
    bob_rewards = [r['walker_2_reward'] for r in results]
    compenetrations = [r['compenetration'] for r in results]

    out_dir = os.path.join(OUTPUT_DIR, reward_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "fixed_meeting_squares.npy"), np.array(meeting_squares))
    np.save(os.path.join(out_dir, "fixed_alice_rewards.npy"), np.array(alice_rewards))
    np.save(os.path.join(out_dir, "fixed_bob_rewards.npy"), np.array(bob_rewards))

    n_comp = sum(compenetrations)
    percentage_comp = n_comp / N_GAMES * 100
    cumulative_alice_reward = np.sum(alice_rewards)
    print(f"[{reward_name}] Compenetrations prevented: {n_comp} ({percentage_comp:.2f}%)")
    print(f"[{reward_name}] Cumulative Alice reward: {cumulative_alice_reward:.2f}")


if __name__ == "__main__":
    # Generate deterministic per-task seeds from the master seed.
    # np.random.seed() only affects the parent process; joblib workers
    # spawn separate processes that don't inherit the parent's random state.
    rng = np.random.RandomState(SEED)
    training_seeds = rng.randint(0, 2**31, size=len(REWARD_FUNCTIONS))
    eval_seeds = {rf.__name__: rng.randint(0, 2**31, size=N_GAMES) for rf in REWARD_FUNCTIONS}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = {
        "n_games": N_GAMES,
        "world_dimension": WORLD_DIMENSION,
        "alice_start": ALICE_START,
        "bob_start": BOB_START,
        "annealing": ANNEALING,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "reward_functions": [f.__name__ for f in REWARD_FUNCTIONS],
    }
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    # Parallelize across the 3 reward functions
    reward_names, alices, bobs = zip(*Parallel(n_jobs=3)(
        delayed(play_with_learning_alice)(rf, seed) for rf, seed in zip(REWARD_FUNCTIONS, training_seeds)
    ))

    # After training, evaluate fixed smart alice against dumb bob for each reward function
    for reward_function, alice, bob in zip(REWARD_FUNCTIONS, alices, bobs):
        playing_with_smart_alice(reward_function, alice, bob, eval_seeds[reward_function.__name__])
