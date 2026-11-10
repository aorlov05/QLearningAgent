import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random


def action_to_text(action):
    """
    Convert action to text for FrozenLake environment.
    """
    if action == 0:
        return "LEFT"
    elif action == 1:
        return "DOWN"
    elif action == 2:
        return "RIGHT"
    elif action == 3:
        return "UP"
    else:
        raise ValueError("Invalid action")


class QLearningAgent:
    NUM_ACTIONS = 4
    """
    Initialize a Q-Learning agent.
    Args:
        Rows, Columns: number of rows and columns in the maze
        Epsilon: Whether to act on policy or randomly
    """
    def __init__(self, rows: int, columns: int, epsilon: float):
        num_of_states = rows * columns
        # Initialize q table to all zeroes
        self.q_table = np.zeros((num_of_states, QLearningAgent.NUM_ACTIONS))
        # Initialize hyperparameters
        self.epsilon = epsilon


# Initialise the environment
# env = gym.make("FrozenLake-v1", render_mode="human",
#                map_name="4x4",
#                desc=generate_random_map(size=4),
#                is_slippery=True,
#                success_rate=0.8,
#                reward_schedule=(10, -10, -.001))
#
# # Reset the environment to generate the first observation
# observation, info = env.reset(seed=42)
# print("start state is ", observation)
# for _ in range(100):
#     # this is where you would insert your policy
#     action = env.action_space.sample()
#     previous_state = observation
#     # step (transition) through the environment with the action
#     # receiving the next observation, reward and if the episode has terminated or truncated
#     observation, reward, terminated, truncated, info = env.step(action)
#     print("Transition (s,a,r,s'): ", previous_state, action_to_text(action), reward, observation)
#     # If the episode has ended then we can reset to start a new episode
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()