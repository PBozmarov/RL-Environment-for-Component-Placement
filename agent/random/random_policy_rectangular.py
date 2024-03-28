"""
Random policy for the rectangular dummy environment.
"""

import random
import numpy as np
from environment.dummy_env_rectangular import DummyPlacementEnv
from typing import Tuple


def random_policy(valid_actions: list) -> Tuple[int, int]:
    """A random policy that returns a random action from the action space.

    Args:
        valid_actions (list): List of valid actions.

    Returns:
        tuple([int,int]): The randomly samplLd action.
    """
    # Choose random action from the valid ones
    action = random.choice(valid_actions)
    return action


def simulate(env: DummyPlacementEnv, policy, n_episodes: int = 1000) -> list:
    """Simulates a policy in the dummy environment.

    Args:
        env (gym.Env): The environment.
        policy (function): The policy
        n_episodes (int): Number of episodes to simulate.

    Returns:
        list: List of episode rewards.
    """
    episode_returns = []  # list of episode returns

    for _ in range(n_episodes):
        episode_return = 0.0
        env.reset()

        # Reset environment
        while True:
            # Get valid actions
            valid_actions = np.argwhere(env.action_mask == 1).tolist()
            # Sample action from policy
            action = policy(valid_actions)
            _, reward, done, _ = env.step(action, verbose=False)
            # Update episode return
            episode_return += reward

            # Check if episode is done
            if done:
                break

        episode_returns.append(episode_return)

    return episode_returns
