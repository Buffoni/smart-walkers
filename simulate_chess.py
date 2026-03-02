import os
import yaml
import numpy as np
import chess
import chess.engine
import random
from collections import Counter, defaultdict
from joblib import Parallel, delayed

# --- Configuration ---
STOCKFISH_PATH = "/usr/games/stockfish"
TOTAL_MOVES = 5000
NUMBER_OF_ITERATIONS = 3
OUTPUT_DIR = "chess_data"
N_JOBS = 16
SEED = 1000
# ---------------------


def simulate_games(skill_level: int, engine_path: str, total_moves: int):
    """Play Stockfish (white) vs random (black) until *total_moves* half-moves."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Skill Level": skill_level})

    fen_history = []
    move_count = 0
    number_of_games = 0

    while move_count < total_moves:
        board = chess.Board()
        fen_history.append(board.board_fen())  # piece placement only
        number_of_games += 1

        while not board.is_game_over() and move_count < total_moves:
            if board.turn == chess.WHITE:
                result = engine.play(board, chess.engine.Limit(time=0.01))
                move = result.move
            else:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            fen_history.append(board.board_fen())
            move_count += 1

    engine.quit()
    return fen_history, number_of_games


def calculate_entropy(fen_list):
    """Shannon entropy (bits) of the empirical FEN distribution."""
    counts = Counter(fen_list)
    total = sum(counts.values())
    entropy = -sum(
        (count / total) * np.log2(count / total) for count in counts.values()
    )
    return entropy, len(counts)


def run_one_task(skill_level, engine_path, total_moves, seed):
    """Worker: simulate games for one (skill, iteration) pair."""
    random.seed(seed)
    fens, number_of_games = simulate_games(skill_level, engine_path, total_moves)
    entropy, unique_positions = calculate_entropy(fens)
    return skill_level, entropy, unique_positions, number_of_games


if __name__ == "__main__":
    rng = np.random.RandomState(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    settings = {
        "stockfish_path": STOCKFISH_PATH,
        "total_moves": TOTAL_MOVES,
        "number_of_iterations": NUMBER_OF_ITERATIONS,
        "seed": SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "settings.yaml"), "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

    # Build list of jobs: (skill, seed) for every (iteration, skill) combo
    tasks = []
    for iteration in range(NUMBER_OF_ITERATIONS):
        for skill in range(21):
            task_seed = int(rng.randint(0, 2**31))
            tasks.append((skill, task_seed))

    print(f"Launching {len(tasks)} tasks with n_jobs={N_JOBS} ...")
    raw_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_one_task)(skill, STOCKFISH_PATH, TOTAL_MOVES, s)
        for skill, s in tasks
    )

    # Collect into per-skill lists
    results = defaultdict(list)
    for skill, entropy, unique_positions, number_of_games in raw_results:
        results[skill].append((entropy, unique_positions, number_of_games))

    # Aggregate across iterations
    skill_levels = np.arange(21)
    entropy_means = np.array([np.mean([e[0] for e in results[s]]) for s in skill_levels])
    entropy_stds = np.array([np.std([e[0] for e in results[s]]) for s in skill_levels])
    unique_means = np.array([np.mean([e[1] for e in results[s]]) for s in skill_levels])
    game_means = np.array([np.mean([e[2] for e in results[s]]) for s in skill_levels])

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, "skill_levels.npy"), skill_levels)
    np.save(os.path.join(OUTPUT_DIR, "entropy_means.npy"), entropy_means)
    np.save(os.path.join(OUTPUT_DIR, "entropy_stds.npy"), entropy_stds)
    np.save(os.path.join(OUTPUT_DIR, "unique_positions_means.npy"), unique_means)
    np.save(os.path.join(OUTPUT_DIR, "game_counts_means.npy"), game_means)

    print(f"\nAll results saved to '{OUTPUT_DIR}/'")
