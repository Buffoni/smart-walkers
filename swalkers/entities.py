import numpy as np
from .functions import softmax

class Walker():
    def __init__(self, name, start_position, world_dimension, use_brain, learn, learning_rate=0.5, discount_factor=1, softmax_temperature=1, greedy=False):
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
        self.learn = learn

        self.greedy = greedy

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
            if self.greedy:
                # Greedy policy
                move = np.argmax(self.q_table[self.position, other_walker_position]) - 1
            else:
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
        if self.learn:
            old_q_value = self.q_table[self.memory['my_previous_position'], self.memory['other_walker_previous_position'], self.memory['my_previous_move'] + 1]
            new_q_value = reward + self.discount_factor * np.max(self.q_table[self.position, other_position])
            self.q_table[self.memory['my_previous_position'], self.memory['other_walker_previous_position'], self.memory['my_previous_move'] + 1] = \
                (1-self.learning_rate) * old_q_value + self.learning_rate * new_q_value
        else:
            # If not learning, we do not update the Q-table
            pass
      
def linear_reward(meeting_square, world_dimension, time=None):
    alice_reward = ((world_dimension-1)/2 - meeting_square)/ ((world_dimension-1)/2)
    bob_reward = -alice_reward
    return alice_reward, bob_reward

def time_dependent_linear_reward(meeting_square, world_dimension, time=0):
    alice_reward = ((world_dimension-1)/2 - meeting_square - time/((world_dimension-4)**2))/ ((world_dimension-1)/2)
    bob_reward = -alice_reward
    return alice_reward, bob_reward

def sinusoidal_reward(meeting_square, world_dimension, time=None):
    # Normalize the meeting square to the range [0, 1]
    normalized_square = meeting_square / (world_dimension - 1)
    alice_reward = np.sin(normalized_square * np.pi)
    bob_reward = -alice_reward
    return alice_reward, bob_reward

def high_frequency_sinusoidal_reward(meeting_square, world_dimension, time=None):
    # Normalize the meeting square to the range [0, 1]
    normalized_square = meeting_square / (world_dimension - 1)
    alice_reward = np.cos(4 * np.pi * normalized_square)
    bob_reward = -alice_reward
    return alice_reward, bob_reward