import numpy as np
from .functions import boltzmann_exploration

class Walker():
    def __init__(self, name, start_position, world_dimension, use_brain, learn, learning_rate=0.5, discount_factor=1, softmax_temperature=1, greedy=False):
        """
        Arguments:
        name: string, name of the walker
        start_position: int, starting position of the walker
        world_dimension: int, dimension of the world (number of squares)
        use_brain: bool, whether the walker uses its brain to choose moves or moves randomly
        learn: bool, whether the walker learns from the rewards or not (different from use_brain!)
        learning_rate: float, learning rate for the Q-learning update
        discount_factor: float, discount factor for the Q-learning update
        softmax_temperature: float, temperature for the Boltzmann exploration
        greedy: bool, whether the walker chooses the move with the highest Q-value or samples from the Boltzmann distribution

        Methods:
        reset(): resets the walker's position and memory to the initial state
        update_memory(): updates the walker's memory with the current position, the other walker's position, and the move taken
        get_policy_tensor(): returns the policy tensor derived from the Q-table using Boltzmann exploration
        choose_move(): chooses a move based on the current policy (either greedy or stochastic)
        check_if_move_is_possible(): checks if the chosen move is possible given the current position and the world boundaries
        move(): updates the walker's position based on the chosen move and the other walker's position
        update_brain(): updates the Q-table based on the received reward and the other walker's position
        
        """
        # Initialization as random walker
        self.q_table = np.ones((world_dimension, world_dimension, 3)) / 3

        self.name = name
        self.start_position = start_position
        self.position = start_position
        self.world_dimension = world_dimension
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.softmax_temperature = softmax_temperature
        self.use_brain = use_brain
        self.learn = learn
        self.greedy = greedy

        self.memory = {
            'my_previous_position': None,
            'my_previous_move': None,
            'other_walker_previous_position': None
        }

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
        return boltzmann_exploration(self.q_table, self.softmax_temperature, axis=-1)

    def choose_move(self, other_walker_position):
        if not self.use_brain:
            return np.random.choice([-1, 0, 1])
        if self.greedy:
            # Greedy policy
            move = np.argmax(self.q_table[self.position, other_walker_position]) - 1
        else:
            # Policy
            probabilities = boltzmann_exploration(self.q_table[self.position, other_walker_position], self.softmax_temperature)
            move = np.random.choice([-1, 0, 1], p=probabilities)    
        return move

    def check_if_move_is_possible(self, move):
        # Check if we end up under 0
        if self.position + move < 0:
            return False
        # Check if we end up over the world dimension
        if self.position + move >= self.world_dimension:
            return False
        return True

    def move(self, other_position):
        move = self.choose_move(other_position)
        self.update_memory(self.position, other_position, move)
        if self.check_if_move_is_possible(move):
            self.position += move

    def update_brain(self, reward, other_position):
        if not self.learn:
            return
        
        old_q_value = self.q_table[
            self.memory['my_previous_position'],
            self.memory['other_walker_previous_position'],
            self.memory['my_previous_move'] + 1 # +1 because moves are -1, 0, 1 and we need shift to 0, 1, 2 for indexing
            ]
        
        new_q_value = reward + self.discount_factor * np.max(self.q_table[self.position, other_position])

        self.q_table[
            self.memory['my_previous_position'],
            self.memory['other_walker_previous_position'],
            self.memory['my_previous_move'] + 1
            ] = (1-self.learning_rate) * old_q_value + self.learning_rate * new_q_value