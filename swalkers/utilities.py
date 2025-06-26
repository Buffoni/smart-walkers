import numpy as np

def check_same_position(walker_1, walker_2):
  if walker_1.position == walker_2.position:
    return True
  else:
    return False

def check_compenetration(walker_1, walker_2):
  if walker_1.memory['my_previous_position'] == walker_2.position and walker_2.memory['my_previous_position'] == walker_1.position:
    return True
  else:
    return False

def check_end_game(walker_1, walker_2):
  end_game = False
  compenetration = False
  if check_same_position(walker_1, walker_2):
    end_game = True
  elif check_compenetration(walker_1, walker_2):
    # Coin toss to choose the final position of the two walkers
    if np.random.rand() < 0.5:
      walker_1.position = walker_2.position
    else:
      walker_2.position = walker_1.position
    end_game = True
    compenetration = True
  else:
    end_game = False

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

          elif j == i + 1 and j_prime == i_prime +1:
            A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 0]

          elif j == i + 1 and j_prime == i_prime:
            A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 1]

          elif j == i + 1 and j_prime == i_prime - 1:
            A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 0] * walker_2_prob_tensor[j_prime, j, 2]

          elif j == i and j_prime == i_prime + 1:
            A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 1] * walker_2_prob_tensor[j_prime, j, 0]

          elif j == i and j_prime == i_prime:
            A[global_i, global_j] = walker_1_prob_tensor[j, j_prime, 1] * walker_2_prob_tensor[j_prime, j, 1]

          elif j == i and j_prime == i_prime -1:
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
  # Insert interactions by replacing the columns with index: i + (i-1)*number_of_sites -1
  # with the identity column
  # (Notice the -1, it is there because the index start from 0 and not from 1)

    number_of_sites = int(np.sqrt(A.shape[0]))

    trap_indices_list = [i + (i-1)*number_of_sites - 1 for i in range(1, number_of_sites+1)]

    for index in trap_indices_list:
        A[:, index] = np.eye(number_of_sites**2)[:, index]

def remove_forbidden_compenetraion_processes(A):
  A_non_comp = np.copy(A)
  number_of_sites = np.sqrt(A.shape[0]).astype(int)

  # Finding indexes of forbidden processes

  indices_i_switch_processes = []
  indices_j_switch_processes = []

  for k in range(1, number_of_sites):
    current_i = k+1+(k-1)*number_of_sites
    current_j = k+k*number_of_sites
    indices_i_switch_processes.append(current_i)
    indices_j_switch_processes.append(current_j)

  for k in range(2, number_of_sites+1):
    current_i = k-1+(k-1)*number_of_sites
    current_j = k+(k-2)*number_of_sites
    indices_i_switch_processes.append(current_i)
    indices_j_switch_processes.append(current_j)

  current_i = 1+(number_of_sites-1)*number_of_sites
  current_j = number_of_sites
  indices_i_switch_processes.append(current_i)
  indices_j_switch_processes.append(current_j)

  current_i = number_of_sites
  current_j = 1+(number_of_sites-1)*number_of_sites
  indices_i_switch_processes.append(current_i)
  indices_j_switch_processes.append(current_j)

  number_of_transfers_needed = len(indices_i_switch_processes)

  # -1 slide to conform to cs indices

  for i in range(number_of_transfers_needed):
    indices_i_switch_processes[i] -= 1
    indices_j_switch_processes[i] -= 1

  indices_i_still_processes = indices_j_switch_processes.copy()
  indices_j_still_processes = indices_j_switch_processes.copy()

  # Perform the transfer of probability mass

  for i in range(number_of_transfers_needed):

    A_non_comp[indices_i_still_processes[i], indices_j_still_processes[i]] += \
      A_non_comp[indices_i_switch_processes[i], indices_j_switch_processes[i]]

    A_non_comp[indices_i_switch_processes[i], indices_j_switch_processes[i]] = 0

  return A_non_comp