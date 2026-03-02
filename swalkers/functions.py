import numpy as np
from scipy.special import softmax
from .utilities import impose_reflective_boundary_conditions, build_matrix_A, add_absorbent_interactions, remove_forbidden_compenetration_processes

def boltzmann_exploration(tensor, temperature, axis=None):
    if temperature <= 0:
        raise Exception('Temperature must be positive')
    return softmax(tensor / temperature, axis=axis)

def calculate_entropy_of_policy_tensor(policy_tensor):
    """
    Compute the normalized Shannon entropy of a policy tensor.

    The total entropy is the sum of per-state Shannon entropies over all states:

        H = sum_{i,j} H(i,j) = -sum_{i,j} sum_k p_{i,j,k} * log2(p_{i,j,k})

    This is normalized by the maximum achievable value.
    Note that: we use np.where to handle numerical instabilities when p_{i,j,k} closes to zero: we treat 0 * log2(0) as 0, which is consistent with the limit definition of entropy.
    """
    norm_constant = np.log2(policy_tensor.shape[2]) * (policy_tensor.shape[0] ** 2)
    entropy = -np.sum(np.where(policy_tensor > 0, policy_tensor * np.log2(policy_tensor), 0.0))
    return entropy / norm_constant

def calculate_entropy_of_A(A):
  """
  Computes the entropy of the system by leveraging the eigenvector corresponding to eigenvalue of 1 in the transition matrix A.
  The entropy is then calculated from the stationary distribution represented by this eigenvector.
  The entropy is also normalized.
  """
  # Calculate the normalization constant
  norm_constant = np.log2(A.shape[0])  

  # Find eigenvector with corresponding eigenvalue 1
  eigenvalues, eigenvectors = np.linalg.eig(A)

  eigenvector = None
  for i in range(len(eigenvalues)):
    if np.isreal(eigenvalues[i]) and abs(eigenvalues[i] - 1) < 1e-10: # Check if the eigenvalue is real and close to 1
      eigenvector = eigenvectors[:, i]
      break
  if eigenvector is None:
    raise Exception('Eigenvector with eigenvalue 1 not found')
  
  # Normalize the eigenvector
  eigenvector = np.real(eigenvector)
  eigenvector = eigenvector / np.sum(eigenvector)

  # Calculate the entropy (with usual instability handling for p*log(p) when p is close to 0)
  entropy = -np.sum(np.where(eigenvector > 0, eigenvector * np.log2(eigenvector), 0.0))

  # Normalize the entropies
  entropy /= norm_constant

  return entropy

def calculate_starting_distribution_in_tensorspace(walker_1_start_pos, walker_2_start_pos, world_dimension):
    p0_walker_1 = np.zeros(world_dimension)
    p0_walker_1[walker_1_start_pos] = 1
    p0_walker_2 = np.zeros(world_dimension)
    p0_walker_2[walker_2_start_pos] = 1
    P0 = np.kron(p0_walker_1, p0_walker_2)
    return P0

def calculate_first_encounter_probabilities(starting_distribution, walker_1_policy_tensor, walker_2_policy_tensor):
    number_of_squares = int(walker_1_policy_tensor.shape[0])
    walker_1_policy_tensor = np.copy(walker_1_policy_tensor)
    walker_2_policy_tensor = np.copy(walker_2_policy_tensor)
    impose_reflective_boundary_conditions(walker_1_policy_tensor)
    impose_reflective_boundary_conditions(walker_2_policy_tensor)
    A = build_matrix_A(walker_1_policy_tensor, walker_2_policy_tensor)
    add_absorbent_interactions(A)
    A = remove_forbidden_compenetration_processes(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Ensure the first N eigenvalues correspond to the absorbing states (eigenvalue = 1).
    # If np.linalg.eig returned them in a different order, reorder so unit eigenvalues come first.
    unity_mask = np.isreal(eigenvalues) & (np.abs(eigenvalues - 1.0) < 1e-10)
    if not np.all(unity_mask[:number_of_squares]):
        unity_idx = np.where(unity_mask)[0]
        other_idx = np.where(~unity_mask)[0]
        new_order = np.concatenate([unity_idx, other_idx])
        eigenvalues = eigenvalues[new_order]
        eigenvectors = eigenvectors[:, new_order]

    M = np.linalg.inv(eigenvectors).T
    M = np.real(M)
    p=[]
    for j in range(number_of_squares):
        prob = 0
        for k in range(number_of_squares**2):
            prob += starting_distribution[k] * M[k, j]
        p.append(prob)
    return p

def calculate_first_encounter_times(A):
    """
    Calculate the expected time to first encounter for each possible starting configuration of the two walkers;
    using our new formula.
    """
    number_of_squares = int(np.sqrt(A.shape[0]))
    A_modded = np.copy(A)
    A_modded = remove_forbidden_compenetration_processes(A_modded)
    trap_indices_list = [i + (i-1)*number_of_squares - 1 for i in range(1, number_of_squares+1)]
    for index in trap_indices_list:
        A_modded[:, index] = np.zeros(number_of_squares**2)
    c_vector = np.ones(number_of_squares**2)
    for index in trap_indices_list:
        c_vector[index] = 0
    main_matrix = np.linalg.inv(np.eye(number_of_squares**2) - np.transpose(A_modded))
    time_vector =np.matmul(main_matrix, c_vector)

    # From the time vector get the time matrix
    time_matrix = np.zeros((number_of_squares, number_of_squares))
    def get_correct_index(a, b):
        return number_of_squares*b + a
    for i in range(number_of_squares):
        for j in range(number_of_squares):
            index = get_correct_index(i, j)
            time_matrix[i, j] = time_vector[index]

    return time_vector, time_matrix