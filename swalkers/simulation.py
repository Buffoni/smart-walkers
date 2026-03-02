from tqdm.auto import tqdm
from .utilities import check_end_game, impose_reflective_boundary_conditions, build_matrix_A, remove_forbidden_compenetration_processes
from .functions import calculate_entropy_of_policy_tensor, calculate_entropy_of_A
import numpy as np

def play_one_game(walker_1, walker_2, world_dimension, reward_function):
    # Reset the walkers
    walker_1.reset()
    walker_2.reset()
    moves = 0
    while True:
        walker_1.move(walker_2.position)
        walker_2.move(walker_1.position)
        moves += 1

        end_game, compenetration = check_end_game(walker_1, walker_2)

        if end_game:
            if walker_1.position == walker_2.position:
                meeting_square = walker_1.position
            else:
                raise Exception('The walkers are not in the same position at the end of the game')

            walker_1_reward, walker_2_reward = reward_function(
                meeting_square,
                world_dimension,
                time = moves
            )

            walker_1.update_brain(walker_1_reward, walker_2.position)
            walker_2.update_brain(walker_2_reward, walker_1.position)
            break
        else:
            walker_1.update_brain(0, walker_2.position)
            walker_2.update_brain(0, walker_1.position)

    game_results = {
        'walker_1_reward': walker_1_reward,
        'walker_2_reward': walker_2_reward,
        'moves': moves,
        'meeting_square': meeting_square,
        'compenetration': compenetration
    }

    return game_results

def playgames(walker_1, walker_2, world_dimension, reward_function, number_of_games, annealing=False, tqdm_position=0, desc=None):
    policy_entropy_list = []
    A_entropy_list = []
    meeting_squares = []
    number_of_compenetrations_prevented = 0
    walker_1_rewards = []
    walker_2_rewards = []

    starting_softmax_temperature = walker_1.softmax_temperature

    for game_number in tqdm(range(number_of_games), desc=desc or walker_1.name, position=tqdm_position, leave=True):

        policy_tensor = walker_1.get_policy_tensor()
        policy_entropy = calculate_entropy_of_policy_tensor(policy_tensor)
        policy_entropy_list.append(policy_entropy)

        policy_tensor_1 = walker_1.get_policy_tensor()
        policy_tensor_2 = walker_2.get_policy_tensor()
        impose_reflective_boundary_conditions(policy_tensor_1)
        impose_reflective_boundary_conditions(policy_tensor_2)
        A = build_matrix_A(policy_tensor_1, policy_tensor_2)
        A = remove_forbidden_compenetration_processes(A)

        A_entropy = calculate_entropy_of_A(A)
        A_entropy_list.append(A_entropy)

        if annealing:
            # Annealing the softmax temperature using the game number and number of games
            walker_1.softmax_temperature = max(0.1 , starting_softmax_temperature * (1 - game_number / number_of_games))

        game_results = play_one_game(
            walker_1, walker_2, world_dimension, reward_function
        )

        if game_results['compenetration']:
            number_of_compenetrations_prevented += 1

        meeting_squares.append(game_results['meeting_square'])
        walker_1_rewards.append(game_results['walker_1_reward'])
        walker_2_rewards.append(game_results['walker_2_reward'])

    games_results = {
        'policy_entropy_list': policy_entropy_list,
        'A_entropy_list': A_entropy_list,
        'meeting_squares': meeting_squares,
        'number_of_compenetrations_prevented': number_of_compenetrations_prevented,
        'walker_1_rewards': walker_1_rewards,
        'walker_2_rewards': walker_2_rewards
    }

    return games_results