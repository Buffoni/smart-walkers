from tqdm.auto import tqdm
from .utilities import check_end_game, impose_reflective_boundary_conditions, build_matrix_A, remove_forbidden_compenetration_processes
from .functions import calculate_entropy_of_policy_tensor, calculate_entropy_of_A

def play_one_game(walker_1, walker_2, world_dimension, reward_function):
    walker_1.reset()
    walker_2.reset()
    moves = 0
    while True:
        # Simultaneous moves: both walkers choose their move based on
        # the other's current position, then both apply at once.
        # This matches the simultaneous-move model assumed by the A matrix.
        walker_1.choose_and_remember_move(walker_2.position)
        walker_2.choose_and_remember_move(walker_1.position)
        walker_1.apply_move()
        walker_2.apply_move()
        moves += 1
        end_game, compenetration = check_end_game(walker_1, walker_2)

        if not end_game:
            walker_1.update_brain(0, walker_2.position)
            walker_2.update_brain(0, walker_1.position)
        else:
            walker_1_reward, walker_2_reward = reward_function(
                walker_1.position, # Equal to walker_2.position since end_game is True
                world_dimension,
                time = moves
            )
            walker_1.update_brain(walker_1_reward, walker_2.position, terminal=True)
            walker_2.update_brain(walker_2_reward, walker_1.position, terminal=True)
            break

    return {
        'walker_1_reward': walker_1_reward,
        'walker_2_reward': walker_2_reward,
        'moves': moves,
        'meeting_square': walker_1.position,
        'compenetration': compenetration
    }

def playgames(walker_1, walker_2, world_dimension, reward_function, number_of_games, annealing=False, tqdm_position=0, desc=None):
    number_of_compenetrations_prevented = 0
    policy_entropy_list = []
    walker_2_policy_entropy_list = []
    A_entropy_list = []
    meeting_squares = []
    walker_1_rewards = []
    walker_2_rewards = []

    starting_softmax_temperature_1 = walker_1.softmax_temperature
    starting_softmax_temperature_2 = walker_2.softmax_temperature

    for game_number in tqdm(range(number_of_games), desc=desc, position=tqdm_position, leave=True):
        # Record the policy entropy and A matrix entropy before the game starts
        policy_tensor_1 = walker_1.get_policy_tensor()
        policy_entropy = calculate_entropy_of_policy_tensor(policy_tensor_1)
        policy_entropy_list.append(policy_entropy)
        policy_tensor_2 = walker_2.get_policy_tensor()
        walker_2_policy_entropy = calculate_entropy_of_policy_tensor(policy_tensor_2)
        walker_2_policy_entropy_list.append(walker_2_policy_entropy)
        impose_reflective_boundary_conditions(policy_tensor_1)
        impose_reflective_boundary_conditions(policy_tensor_2)
        A = build_matrix_A(policy_tensor_1, policy_tensor_2)
        A = remove_forbidden_compenetration_processes(A)
        A_entropy = calculate_entropy_of_A(A)
        A_entropy_list.append(A_entropy)

        if annealing:
            decay = max(0.1, 1 - game_number / number_of_games)
            walker_1.softmax_temperature = starting_softmax_temperature_1 * decay
            walker_2.softmax_temperature = starting_softmax_temperature_2 * decay

        game_results = play_one_game(
            walker_1, walker_2, world_dimension, reward_function
        )

        if game_results['compenetration']:
            number_of_compenetrations_prevented += 1

        meeting_squares.append(game_results['meeting_square'])
        walker_1_rewards.append(game_results['walker_1_reward'])
        walker_2_rewards.append(game_results['walker_2_reward'])

    return {
        'policy_entropy_list': policy_entropy_list,
        'walker_2_policy_entropy_list': walker_2_policy_entropy_list,
        'A_entropy_list': A_entropy_list,
        'meeting_squares': meeting_squares,
        'number_of_compenetrations_prevented': number_of_compenetrations_prevented,
        'walker_1_rewards': walker_1_rewards,
        'walker_2_rewards': walker_2_rewards
    }