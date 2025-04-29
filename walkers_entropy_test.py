import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

NUMBER_OF_SQUARES = 11
NUMBER_OF_GAMES = int(1e2)
START_POS_ALICE = 1
START_POS_BOB = NUMBER_OF_SQUARES - 2

GAMES_FOR_ENTROPY_OF_POSITION_CALCULATION = NUMBER_OF_GAMES
PLOT_NEGENTROPY = False

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=100)

#----------------------------------------------------------------------------------

def softmax(vector, temperature):
  exp_vector = np.exp(vector / temperature)
  return exp_vector / np.sum(exp_vector)

#----------------------------------------------------------------------------------

class Walker():
  def __init__(self, name, start_position, world_dimension, use_brain, learning_rate=0.5, discount_factor=1, softmax_temperature=1):
    self.name = name
    self.start_position = start_position
    self.position = start_position

    self.memory = {
        'my_previous_position': None,
        'my_previous_move': None,
        'other_walker_previous_position': None
    }

    self.world_dimension = world_dimension

    # Initialization as random walker
    self.q_table = np.ones((world_dimension, world_dimension, 3)) / 3

    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

    self.softmax_temperature = softmax_temperature

    self.use_brain = use_brain

  def reset(self):
    self.position = self.start_position
    self.memory = {
        'my_previous_position': None,
        'my_previous_move': None,
        'other_walker_previous_position': None
    }

  def update_memory(self, my_position, other_position, my_move):
    self.memory['my_previous_position'] = my_position
    self.memory['my_previous_move'] = my_move
    self.memory['other_walker_previous_position'] = other_position

  def get_policy_tensor(self):
    probability_tensor = np.zeros((self.world_dimension, self.world_dimension, 3))
    for i in range(self.world_dimension):
      for j in range(self.world_dimension):
        probability_tensor[i, j] = softmax(self.q_table[i, j], self.softmax_temperature)
    return probability_tensor

  def choose_move(self, other_walker_position):
    if self.use_brain:
      # Policy
      probabilities = softmax(self.q_table[self.position, other_walker_position], self.softmax_temperature)
      move = np.random.choice([-1, 0, 1], p=probabilities)
    else:
      move = np.random.choice([-1, 0, 1])
    return move

  def check_if_move_is_possible(self, move):
    # Check if we end up under 0
    if self.position + move < 0:
      return False
    # Check if we end up over the world dimension
    elif self.position + move >= self.world_dimension:
      return False
    else:
      return True

  def move(self, other_position):
    move = self.choose_move(other_position)
    self.update_memory(self.position, other_position, move)
    if self.check_if_move_is_possible(move):
      self.position += move

  def update_brain(self, reward, other_position):
    if self.use_brain:
      old_q_value = self.q_table[self.memory['my_previous_position'], self.memory['other_walker_previous_position'], self.memory['my_previous_move'] + 1]
      new_q_value = reward + self.discount_factor * np.max(self.q_table[self.position, other_position])
      self.q_table[self.memory['my_previous_position'], self.memory['other_walker_previous_position'], self.memory['my_previous_move'] + 1] = \
        (1-self.learning_rate) * old_q_value + self.learning_rate * new_q_value
      
#-----------------------------------------------------------------------------------

def check_same_position(walker_1, walker_2):
  if walker_1.position == walker_2.position:
    return True
  else:
    return False
  
#-----------------------------------------------------------------------------------

def check_compenetration(walker_1, walker_2):
  if walker_1.memory['my_previous_position'] == walker_2.position and walker_2.memory['my_previous_position'] == walker_1.position:
    return True
  else:
    return False
  
#-----------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------

def calculate_reward(meeting_square, world_dimension):
  walker_1_reward = ((world_dimension - 1)/ 2) - meeting_square
  walker_2_reward = meeting_square - ((world_dimension - 1) / 2)
  return walker_1_reward, walker_2_reward

#-----------------------------------------------------------------------------------

def play_one_game(walker_1, walker_2, world_dimension, learning=True):
  # Reset the walkers
  walker_1.reset()
  walker_2.reset()
  moves = 0
  walker_1_positions = []
  walker_2_positions = []
  while True:
    walker_1_positions.append(walker_1.position)
    walker_2_positions.append(walker_2.position)
    walker_1.move(walker_2.position)
    walker_2.move(walker_1.position)
    moves += 1

    end_game, compenetration = check_end_game(walker_1, walker_2)

    if end_game:

      if walker_1.position != walker_2.position:
        raise Exception('The walkers are not in the same position at the end of the game')
      else:
        meeting_square = walker_1.position

      walker_1_reward, walker_2_reward = calculate_reward(
          meeting_square,
          world_dimension,
        )
      if learning:
        walker_1.update_brain(walker_1_reward, walker_2.position)
        walker_2.update_brain(walker_2_reward, walker_1.position)
      break
    else:
      if learning:
        walker_1.update_brain(0, walker_2.position)
        walker_2.update_brain(0, walker_1.position)

  return walker_1_reward, walker_2_reward, moves, meeting_square, walker_1_positions, walker_2_positions

#-----------------------------------------------------------------------------------

def calculate_entropy_of_positions(walker_positions):
  probs = np.zeros(NUMBER_OF_SQUARES)

  for i in range(NUMBER_OF_SQUARES):
    probs[i] = walker_positions.count(i) / len(walker_positions)

  entropy = -np.sum(probs * np.log(probs + 1e-10))
  negentropy = np.log(NUMBER_OF_SQUARES + 1e-10) - entropy

  return entropy, negentropy

#-----------------------------------------------------------------------------------

def calculate_entropy_of_probability_tensor(probability_tensor):
  entropy = 0
  for i in range(probability_tensor.shape[0]):
    for j in range(probability_tensor.shape[1]):
      for k in range(probability_tensor.shape[2]):
        entropy -= probability_tensor[i, j, k] * np.log(probability_tensor[i, j, k] + 1e-10)

  maximum_possible_entropy = np.log(probability_tensor.shape[2] + 1e-10) * probability_tensor.shape[0] * probability_tensor.shape[1]
  negentropy = maximum_possible_entropy - entropy

  return entropy, negentropy

#-----------------------------------------------------------------------------------

def convert_indices_to_global(i, j, i_prime, j_prime):
    '''
    This is crucial!
    (i, j) position and future position of the first walker
    (i_prime, j_prime) position and future position of the second walker
    The priming indicates the second walker!
    ATTENTION: the global indices work only if swapped for some reason!
    '''
    global_i = i * NUMBER_OF_SQUARES + i_prime
    global_j = j * NUMBER_OF_SQUARES + j_prime
    return global_i, global_j

#-----------------------------------------------------------------------------------

def impose_reflective_boundary_conditions(probability_tensor):
  '''
  Transfers the probability mass from the sides to the stationary state.
  This is crucial since in the simulation the walkers are not allowed to go outside the world.
  And this limit is enforced by the environment!
  '''
  for i in range(NUMBER_OF_SQUARES):
    probability_tensor[0, i, 1] += probability_tensor[0, i, 0]
    probability_tensor[0, i, 0] = 0
    probability_tensor[NUMBER_OF_SQUARES-1, i, 1] += probability_tensor[NUMBER_OF_SQUARES-1, i, 2]
    probability_tensor[NUMBER_OF_SQUARES-1, i, 2] = 0

#-----------------------------------------------------------------------------------

def build_matrix_A(walker_1_prob_tensor, walker_2_prob_tensor):
  '''
  Be careful! j (j') is the starting position!
  '''

  A = np.zeros((NUMBER_OF_SQUARES**2, NUMBER_OF_SQUARES**2))

  for i in range(NUMBER_OF_SQUARES):
    for j in range(NUMBER_OF_SQUARES):
      for i_prime in range(NUMBER_OF_SQUARES):
        for j_prime in range(NUMBER_OF_SQUARES):
          
          global_i, global_j = convert_indices_to_global(i, j, i_prime, j_prime)

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

#-----------------------------------------------------------------------------------

def calculate_entropy_of_A(A):
  # Find eigenvector with corresponding eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(A)
    index = np.where(np.isclose(eigenvalues, 1))[0][0]
    eigenvector = eigenvectors[:, index]
    eigenvector = np.real(eigenvector)
    probs = np.zeros(NUMBER_OF_SQUARES**2)
    entropy = -np.sum(eigenvector * np.log(eigenvector + 1e-10))
    negentropy = np.log(NUMBER_OF_SQUARES**2 + 1e-10) - entropy
    return entropy, negentropy

#-----------------------------------------------------------------------------------
# MAIN

if __name__ == '__main__':
  # Initialize the walkers
  alice = Walker('Alice', START_POS_ALICE, NUMBER_OF_SQUARES, use_brain=True)
  bob = Walker('Bob', START_POS_BOB, NUMBER_OF_SQUARES, use_brain=True)

  # Start training loop

  alice_entropy_of_tensor = []
  alice_negentropy_of_tensor = []
  alice_entropy_of_A = []
  alice_negentropy_of_A = []
  alice_entropy_of_position = []
  alice_negentropy_of_position = []
  
  alice_cumulative_rewards = []
  bob_cumulative_rewards = []
  alice_cumulative_rewards.append(0)
  bob_cumulative_rewards.append(0)

  for _ in tqdm(range(NUMBER_OF_GAMES)):
    # Play one game
    alice_reward, bob_reward, _,_,_,_ = play_one_game(alice, bob, NUMBER_OF_SQUARES)
    alice_cumulative_rewards.append(alice_cumulative_rewards[-1] + alice_reward)
    bob_cumulative_rewards.append(bob_cumulative_rewards[-1] + bob_reward)

    # Calculate the probability tensor

    alice_probability_tensor = alice.get_policy_tensor()
    bob_probability_tensor = bob.get_policy_tensor()

    impose_reflective_boundary_conditions(alice_probability_tensor)
    impose_reflective_boundary_conditions(bob_probability_tensor)

    # Calculate the entropy of the probability tensor
    alice_entropy, alice_negentropy = calculate_entropy_of_probability_tensor(alice_probability_tensor)
    alice_entropy_of_tensor.append(alice_entropy)
    alice_negentropy_of_tensor.append(alice_negentropy)

    # Build the matrix A

    A = build_matrix_A(alice_probability_tensor, bob_probability_tensor)

    # Calculate the entropy of A

    A_entropy, A_negentropy = calculate_entropy_of_A(A)
    alice_entropy_of_A.append(A_entropy)
    alice_negentropy_of_A.append(A_negentropy)

    # Calculate the entropy of the positions
    concat_alice_positions = []

    for _ in range(GAMES_FOR_ENTROPY_OF_POSITION_CALCULATION):
      _,_,_,_, alice_positions, _ = play_one_game(alice, bob, NUMBER_OF_SQUARES, learning=False)
      concat_alice_positions.extend(alice_positions)
    
    walker_1_entropy, walker_1_negentropy = calculate_entropy_of_positions(concat_alice_positions)
    alice_entropy_of_position.append(walker_1_entropy)
    alice_negentropy_of_position.append(walker_1_negentropy)

  #Plot the results
  if PLOT_NEGENTROPY:
    plt.plot(alice_negentropy_of_tensor, label='Negentropy of tensor')
    plt.plot(alice_negentropy_of_A, label='Negentropy of A')
    plt.plot(alice_negentropy_of_position, label='Negentropy of position')

  plt.plot(alice_entropy_of_tensor, label='Entropy of tensor')
  plt.plot(alice_entropy_of_A, label='Entropy of A')
  plt.plot(alice_entropy_of_position, label='Entropy of position')

  plt.xlabel('Games')
  plt.ylabel('Entropy')
  plt.title('Entropy of the walkers')
  plt.legend()
  plt.savefig('entropy_of_walkers.png')

  plt.clf()

  plt.plot(alice_cumulative_rewards, label='Alice cumulative rewards')
  plt.plot(bob_cumulative_rewards, label='Bob cumulative rewards')
  plt.xlabel('Games')
  plt.ylabel('Cumulative rewards')
  plt.title('Cumulative rewards of the walkers')
  plt.legend()
  plt.savefig('cumulative_rewards_of_walkers.png')
  
  print('Done!')