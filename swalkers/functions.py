import numpy as np
from .utilities import impose_reflective_boundary_conditions, build_matrix_A, add_absorbent_interactions, remove_forbidden_compenetration_processes

def softmax(vector, temperature):
    if temperature <= 0:
        raise Exception('Temperature must be positive')
    exp_vector = np.exp(vector / temperature)
    return exp_vector / np.sum(exp_vector)

def calculate_entropy_of_policy_tensor(policy_tensor):
    # Calculate the normalization constant
    norm_constant = np.log2(policy_tensor.shape[2])* (policy_tensor.shape[0]**2)

    entropy = 0
    for i in range(policy_tensor.shape[0]):
        for j in range(policy_tensor.shape[1]):
            for k in range(policy_tensor.shape[2]):
                entropy -= policy_tensor[i, j, k] * np.log2(policy_tensor[i, j, k])

    entropy /= norm_constant

    return entropy

def calculate_entropy_of_A(A):
  # Calculate the normalization constant
  norm_constant = np.log2(A.shape[0])  

  # Find eigenvector with corresponding eigenvalue 1
  eigenvalues, eigenvectors = np.linalg.eig(A)

  eigenvector = None
  
  for i in range(len(eigenvalues)):
    if np.isreal(eigenvalues[i]) and eigenvalues[i] - 1 < 1e-10:
      eigenvector = eigenvectors[:, i]
      break
  
  if eigenvector is None:
    raise Exception('Eigenvector with eigenvalue 1 not found')
  
  # Normalize the eigenvector
  eigenvector = np.real(eigenvector)
  eigenvector = eigenvector / np.sum(eigenvector)

  # Calculate the entropy
  entropy = -np.sum(eigenvector * np.log2(eigenvector + 1e-10))

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

    impose_reflective_boundary_conditions(walker_1_policy_tensor)
    impose_reflective_boundary_conditions(walker_2_policy_tensor)

    A = build_matrix_A(walker_1_policy_tensor, walker_2_policy_tensor)
    add_absorbent_interactions(A)
    A = remove_forbidden_compenetration_processes(A)

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:", eigenvalues)

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

    # Calculate the time vector
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