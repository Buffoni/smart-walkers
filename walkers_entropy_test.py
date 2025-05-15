import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

NUMBER_OF_SQUARES = 11
NUMBER_OF_GAMES = int(1e3)
START_POS_ALICE = 1
START_POS_BOB = NUMBER_OF_SQUARES - 2

GAMES_FOR_ENTROPY_OF_POSITION_CALCULATION = int(1e2)
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
  # First we calculate the normalization constant

  norm_constant = np.log(NUMBER_OF_SQUARES)

  # Then we calculate the entropies
  probs = np.zeros(NUMBER_OF_SQUARES)

  for i in range(NUMBER_OF_SQUARES):
    probs[i] = walker_positions.count(i) / len(walker_positions)

  entropy = -np.sum(probs * np.log(probs + 1e-10))
  negentropy = np.log(NUMBER_OF_SQUARES + 1e-10) - entropy

  # Normalize the entropy
  entropy /= norm_constant
  negentropy /= norm_constant

  return entropy, negentropy

#-----------------------------------------------------------------------------------

def calculate_entropy_of_probability_tensor(probability_tensor):
  # First we calculate the normalization constant
  norm_constant = np.log(probability_tensor.shape[2])*probability_tensor.shape[0]**2

  # Then we calculate the entropies
  entropy = 0
  for i in range(probability_tensor.shape[0]):
    for j in range(probability_tensor.shape[1]):
      for k in range(probability_tensor.shape[2]):
        entropy -= probability_tensor[i, j, k] * np.log(probability_tensor[i, j, k])

  negentropy = norm_constant - entropy

  # Normalize the entropy
  entropy /= norm_constant
  negentropy /= norm_constant

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
  # Calculate the normalization constant
  norm_constant = np.log(A.shape[0])  

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
  entropy = -np.sum(eigenvector * np.log(eigenvector + 1e-10))

  negentropy = np.log(A.shape[0] + 1e-10) - entropy

  # Normalize the entropies
  entropy /= norm_constant
  negentropy /= norm_constant

  return entropy, negentropy

#-----------------------------------------------------------------------------------
# MAIN

if __name__ == '__main__':

  # Create output directory if it does not exist
  # Empty the directory if it exists

  if os.path.exists('Entropy_Outputs'):
    for file in os.listdir('Entropy_Outputs'):
      os.remove(os.path.join('Entropy_Outputs', file))
  else:
    os.mkdir('Entropy_Outputs')
    
  # Initialize the walkers
  alice = Walker('Alice', START_POS_ALICE, NUMBER_OF_SQUARES, use_brain=True)
  bob = Walker('Bob', START_POS_BOB, NUMBER_OF_SQUARES, use_brain=False)

  # Start training loop

  alice_entropy_of_tensor = []
  alice_negentropy_of_tensor = []
  alice_entropy_of_position = []
  alice_negentropy_of_position = []

  bob_entropy_of_tensor = []
  bob_negentropy_of_tensor = []
  bob_entropy_of_position = []
  bob_negentropy_of_position = []

  A_entropys = []
  A_negentropys = []

  alice_cumulative_rewards = []
  bob_cumulative_rewards = []
  alice_cumulative_rewards.append(0)
  bob_cumulative_rewards.append(0)

  for _ in tqdm(range(NUMBER_OF_GAMES)):

    # Calculate the probability tensor

    alice_probability_tensor = alice.get_policy_tensor()
    bob_probability_tensor = bob.get_policy_tensor()

    # Calculate the entropy of the probability tensor
    alice_entropy, alice_negentropy = calculate_entropy_of_probability_tensor(alice_probability_tensor)
    bob_entropy, bob_negentropy = calculate_entropy_of_probability_tensor(bob_probability_tensor)
    alice_entropy_of_tensor.append(alice_entropy)
    bob_entropy_of_tensor.append(bob_entropy)
    alice_negentropy_of_tensor.append(alice_negentropy)
    bob_negentropy_of_tensor.append(bob_negentropy)

    # Play one game
    alice_reward, bob_reward, _,_,_,_ = play_one_game(alice, bob, NUMBER_OF_SQUARES, learning=True)
    alice_cumulative_rewards.append(alice_cumulative_rewards[-1] + alice_reward)
    bob_cumulative_rewards.append(bob_cumulative_rewards[-1] + bob_reward)

    impose_reflective_boundary_conditions(alice_probability_tensor)
    impose_reflective_boundary_conditions(bob_probability_tensor)

    # Build the matrix A

    A = build_matrix_A(alice_probability_tensor, bob_probability_tensor)

    # Calculate the entropy of A

    A_entropy, A_negentropy = calculate_entropy_of_A(A)
    A_entropys.append(A_entropy)
    A_negentropys.append(A_negentropy)

    # Calculate the entropy of the positions
    concat_alice_positions = []
    concat_bob_positions = []

    for _ in range(GAMES_FOR_ENTROPY_OF_POSITION_CALCULATION):
      _,_,_,_, alice_positions, bob_positions = play_one_game(alice, bob, NUMBER_OF_SQUARES, learning=False)
      concat_alice_positions.extend(alice_positions)
      concat_bob_positions.extend(bob_positions)
    
    alice_position_entropy, alice_position_negentropy = calculate_entropy_of_positions(concat_alice_positions)
    bob_position_entropy, bob_position_negentropy = calculate_entropy_of_positions(concat_bob_positions)
    alice_entropy_of_position.append(alice_position_entropy)
    bob_entropy_of_position.append(bob_position_entropy)
    alice_negentropy_of_position.append(alice_position_negentropy)
    bob_negentropy_of_position.append(bob_position_negentropy)

  # Make the 3 plots of entropy separately

  plt.plot(alice_entropy_of_tensor, label='Alice entropy of tensor')
  plt.plot(bob_entropy_of_tensor, label='Bob entropy of tensor')
  plt.xlabel('Games')
  plt.ylabel('Entropy of tensor')
  plt.title('Entropy of the tensor of the walkers')
  plt.legend()
  plt.savefig(os.path.join('Entropy_Outputs', 'entropy_of_tensor_of_walkers.png'))

  plt.clf()

  plt.plot(alice_entropy_of_position, label='Alice entropy of position')
  plt.plot(bob_entropy_of_position, label='Bob entropy of position')
  plt.xlabel('Games')
  plt.ylabel('Entropy of position')
  plt.title('Entropy of the position of the walkers')
  plt.legend()
  plt.savefig(os.path.join('Entropy_Outputs', 'entropy_of_position_of_walkers.png'))

  plt.clf()

  plt.plot(A_entropys, label='Entropy of A')
  plt.xlabel('Games')
  plt.ylabel('Entropy of A')
  plt.title('Entropy of A')
  plt.legend()
  plt.savefig(os.path.join('Entropy_Outputs', 'entropy_of_A.png'))

  plt.clf()

  plt.plot(alice_cumulative_rewards, label='Alice cumulative rewards')
  plt.plot(bob_cumulative_rewards, label='Bob cumulative rewards')
  plt.xlabel('Games')
  plt.ylabel('Cumulative rewards')
  plt.title('Cumulative rewards of the walkers')
  plt.legend()
  plt.savefig(os.path.join('Entropy_Outputs', 'cumulative_rewards_of_walkers.png'))
  
  print('Done!')