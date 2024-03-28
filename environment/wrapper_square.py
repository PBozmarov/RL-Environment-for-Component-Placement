"""
Wrappers for the square environment.
"""

import gym  # type: ignore
import numpy as np


class FlatteningActionMaskObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        for v in env.observation_space.spaces.values():
            assert isinstance(v, gym.spaces.Box)
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.utils.flatten_space(v) if k == "action_mask" else v
                for k, v in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation):
        return {
            k: gym.spaces.utils.flatten(self.observation_space[k], v)
            if k == "action_mask"
            else v
            for k, v in observation.items()
        }


class FlatteningActionWrapper(gym.ActionWrapper):
    """A Gym environment action wrapper for the square
    environment.

    Action wrapper which flattens a tuple of discrete actions into
    a single discrete action for the square environment.

    Attributes:
        factor_sizes (list): The number of discrete actions for each
            discrete action space.
    """

    def __init__(self, env):
        """Initializes the action wrapper.

        Args:
            env (gym.Env): The original Gym environment.
        """
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        self.factor_sizes = [space.n for space in env.action_space.spaces]
        self.action_space = gym.spaces.Discrete(np.prod(self.factor_sizes))

    def action(self, action):
        """Translate discrete action into tuple of discrete actions
        for each discrete action space.

        Args:
            action (int): The discrete action.

        Returns:
            list: The list of discrete actions for each discrete action space.
        """
        unflattened_action = []

        action_x, action_y = divmod(action, self.factor_sizes[1])
        unflattened_action.extend([action_x, action_y])

        return unflattened_action

    def validate_action(self, action):
        """Validate the action.

        Args:
            action (int): The action to be validated.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        return self.env.validate_action(*self.action(action))
