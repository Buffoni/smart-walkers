import numpy as np

def check_same_position(walker_1, walker_2):
    if walker_1.position == walker_2.position:
        return True
    return False

def check_compenetration(walker_1, walker_2):
    if walker_1.memory['my_previous_position'] == walker_2.position and walker_2.memory['my_previous_position'] == walker_1.position:
        return True
    return False

def check_end_game(walker_1, walker_2):
    if check_same_position(walker_1, walker_2):
        end_game = True
        compenetration = False
        return end_game, compenetration
    if check_compenetration(walker_1, walker_2):
        # Coin toss to choose the final position of the two walkers
        if np.random.rand() < 0.5:
            walker_1.position = walker_2.position
        else:
            walker_2.position = walker_1.position
        end_game = True
        compenetration = True
        return end_game, compenetration
    end_game = False
    compenetration = False
    return end_game, compenetration

def convert_indices_to_global(i, j, i_prime, j_prime, number_of_squares):
    '''
    This is crucial!
    (i, j) position and future position of the first walker
    (i_prime, j_prime) position and future position of the second walker
    The priming indicates the second walker!
    ATTENTION: the global indices work only if swapped for some reason!
    '''
    global_i = i * number_of_squares + i_prime
    global_j = j * number_of_squares + j_prime
    return global_i, global_j

def impose_reflective_boundary_conditions(probability_tensor):
    '''
    Transfers the probability mass from the sides to the stationary state.
    This is crucial since in the simulation the walkers are not allowed to go outside the world.
    And this limit is enforced by the environment!
    '''
    number_of_squares = probability_tensor.shape[0]
    for i in range(number_of_squares):
        probability_tensor[0, i, 1] += probability_tensor[0, i, 0]
        probability_tensor[0, i, 0] = 0
        probability_tensor[number_of_squares-1, i, 1] += probability_tensor[number_of_squares-1, i, 2]
        probability_tensor[number_of_squares-1, i, 2] = 0

def build_matrix_A(walker_1_prob_tensor, walker_2_prob_tensor):
    '''
    Be careful! j (j') is the starting position!
    '''
    number_of_squares = walker_1_prob_tensor.shape[0]
    A = np.zeros((number_of_squares**2, number_of_squares**2))
    for i in range(number_of_squares):
        for j in range(number_of_squares):
            for i_prime in range(number_of_squares):
                for j_prime in range(number_of_squares):

                    global_i, global_j = convert_indices_to_global(i, j, i_prime, j_prime, number_of_squares)

                    if np.abs(j-i) > 1 or np.abs(j_prime-i_prime) > 1:
                        A[global_i, global_j] = 0

                    elif j == i + 1 and j_prime == i_prime + 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 0]

                    elif j == i + 1 and j_prime == i_prime:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 1]

                    elif j == i + 1 and j_prime == i_prime - 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 2]

                    elif j == i and j_prime == i_prime + 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 1] * walker_2_prob_tensor[j_prime, j, 0]

                    elif j == i and j_prime == i_prime:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 1] * walker_2_prob_tensor[j_prime, j, 1]

                    elif j == i and j_prime == i_prime - 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 1] * walker_2_prob_tensor[j_prime, j, 2]

                    elif j == i - 1 and j_prime == i_prime + 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 2] * walker_2_prob_tensor[j_prime, j, 0]

                    elif j == i - 1 and j_prime == i_prime:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 2] * walker_2_prob_tensor[j_prime, j, 1]

                    elif j == i - 1 and j_prime == i_prime - 1:
                        A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 2] * walker_2_prob_tensor[j_prime, j, 2]

                    else:
                        raise Exception('This should not happen')

    return A

def add_absorbent_interactions(A):
    '''
    Insert interactions by replacing the columns with index: i + (i-1)*number_of_sites -1
    with the identity column
    (Notice the -1, it is there because the index start from 0 and not from 1)
    '''
    number_of_sites = int(np.sqrt(A.shape[0]))
    trap_indices_list = [i + (i-1)*number_of_sites - 1 for i in range(1, number_of_sites+1)]
    for index in trap_indices_list:
        A[:, index] = np.eye(number_of_sites**2)[:, index]

def remove_forbidden_compenetration_processes(A):
    '''
    Removes forbidden "swap" (compenetration) transitions from the transition matrix A
    and redistributes their probability mass to the correct meeting states.

    A compenetration occurs when walker 1 moves from site A to site B while walker 2
    simultaneously moves from site B to site A: they pass through each other, which is
    physically forbidden. In the simulation (see check_end_game), this is resolved
    by a fair coin toss: both walkers end up together at site a or site b with equal
    probability, i.e. it becomes a meeting event.

    In the matrix A, a compenetration is an off-diagonal entry A[i_sw, j_sw] where:
      - j_sw encodes the source configuration (w1 at some site, w2 at adjacent site)
      - i_sw encodes the swapped configuration (they exchanged positions)
    This function:
      1. Identifies all such (i_sw, j_sw) entries using 1-indexed formulas (then shifts to 0-indexed).
      2. For each one, reads the source state j_sw to find the two walker positions:
           w1_pos = j_sw // N,  w2_pos = j_sw % N   (where N = number_of_sites)
      3. Splits the compenetration probability 50/50 into the two meeting states:
           meeting at w1_pos -> global index w1_pos * (N + 1)  i.e. both walkers at w1_pos
           meeting at w2_pos -> global index w2_pos * (N + 1)  i.e. both walkers at w2_pos
      4. Zeros out the original forbidden swap entry.
    '''
    A_non_comp = np.copy(A)
    number_of_sites = np.sqrt(A.shape[0]).astype(int)

    # --- Step 1: Identify forbidden swap transitions (1-indexed formulas) ---
    # These are pairs (i_sw, j_sw) where i_sw is the "swapped" destination
    # and j_sw is the source state with walkers on adjacent sites.

    indices_i_switch_processes = []
    indices_j_switch_processes = []

    # Swaps where walker 1 moves right and walker 2 moves left
    for k in range(1, number_of_sites):
        current_i = k+1+(k-1)*number_of_sites
        current_j = k+k*number_of_sites
        indices_i_switch_processes.append(current_i)
        indices_j_switch_processes.append(current_j)

    # Swaps where walker 1 moves left and walker 2 moves right
    for k in range(2, number_of_sites+1):
        current_i = k-1+(k-1)*number_of_sites
        current_j = k+(k-2)*number_of_sites
        indices_i_switch_processes.append(current_i)
        indices_j_switch_processes.append(current_j)

    # Boundary swap cases
    current_i = 1+(number_of_sites-1)*number_of_sites
    current_j = number_of_sites
    indices_i_switch_processes.append(current_i)
    indices_j_switch_processes.append(current_j)

    current_i = number_of_sites
    current_j = 1+(number_of_sites-1)*number_of_sites
    indices_i_switch_processes.append(current_i)
    indices_j_switch_processes.append(current_j)

    number_of_transfers_needed = len(indices_i_switch_processes)

    # Convert from 1-indexed to 0-indexed
    for i in range(number_of_transfers_needed):
        indices_i_switch_processes[i] -= 1
        indices_j_switch_processes[i] -= 1

    # --- Step 2 & 3: Redistribute probability mass to meeting states ---

    for i in range(number_of_transfers_needed):
        i_sw = indices_i_switch_processes[i]  # destination (forbidden swapped config)
        j_sw = indices_j_switch_processes[i]  # source (adjacent walkers config)
        prob = A_non_comp[i_sw, j_sw]

        # Decode walker positions from the source state index
        w1_pos = j_sw // number_of_sites
        w2_pos = j_sw % number_of_sites

        # Global indices for the two meeting states (both walkers at same site)
        # For site s, the meeting state index is s*N + s = s*(N+1)
        meeting_at_w1 = w1_pos * (number_of_sites + 1)
        meeting_at_w2 = w2_pos * (number_of_sites + 1)

        # Coin toss: 50% probability of meeting at each walker's position
        A_non_comp[meeting_at_w1, j_sw] += 0.5 * prob
        A_non_comp[meeting_at_w2, j_sw] += 0.5 * prob
        A_non_comp[i_sw, j_sw] = 0

    return A_non_comp