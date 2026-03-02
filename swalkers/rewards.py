import numpy as np

def linear_reward(meeting_square, world_dimension, time=None):
    """
    Simple adversarial linear reward, symmetrical and normalized to be between -1 and 1.
    """
    alice_reward = 1 - 2*meeting_square/(world_dimension-1)
    bob_reward = -alice_reward
    return alice_reward, bob_reward

def time_dependent_linear_reward(meeting_square, world_dimension, time=0, lambdav=0.004):
    """
    Time-dependent linear reward that starts as the simple linear reward and then, for Alice, it decreases linearly with time (with a slope of -lambdav).
    """
    alice_reward = 1 - 2*meeting_square/(world_dimension-1) - lambdav * time
    bob_reward = -alice_reward
    return alice_reward, bob_reward

def sinusoidal_reward(meeting_square, world_dimension, time=None):
    """
    Sinusoidal reward, symmetrical and normalized to be between -1 and 1.
    The shape of this reward is really similar to the natural encounter distribution of the 1D walkers starting on opposite sides.
    """
    alice_reward = np.sin(np.pi * meeting_square / (world_dimension - 1))
    bob_reward = -alice_reward
    return alice_reward, bob_reward