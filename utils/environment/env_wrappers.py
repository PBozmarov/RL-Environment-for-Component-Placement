"""
Environment wrappers for the rectangular and square environments.
"""

import gym  # type: ignore
import numpy as np


class FlatteningActionMaskObservationWrapperRect(gym.ObservationWrapper):
    """A Gym environment observation wrapper for the rectangular
    environment.

    This wrapper flattens the action mask in the observation space.

    Attributes:
        observation_space (gym.spaces.Dict): The observation space.
    """

    def __init__(self, env):
        """Initializes the observation wrapper.

        Fllatens the action mask in the observation space.

        Args:
            env (gym.Env): The original Gym environment.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.utils.flatten_space(v) if k == "action_mask" else v
                for k, v in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation):
        """Overrides the default method of the superclass to
        flatten the "action_mask" space in the observation dictionary.

        Args:
            observation (dict): The original observation dictionary.

        Returns:
            dict: The modified observation dictionary with the "action_mask" space
                flattened.
        """
        return {
            k: gym.spaces.utils.flatten(self.observation_space[k], v)
            if k == "action_mask"
            else v
            for k, v in observation.items()
        }


class FlatteningActionWrapperRect(gym.ActionWrapper):
    """A Gym environment action wrapper for the rectangular
    environment.

    Action wrapper which flattens a tuple of discrete actions into
    a single discrete action for the rectangular environment.

    Attributes:
        factor_sizes (list): The list of the sizes of the discrete action spaces.
        action_space (gym.spaces.Discrete): The discrete action space.
    """

    def __init__(self, env):
        """Initializes the action wrapper.

        Args:
            env (gym.Env): The original Gym environment.

        Raises:
            AssertionError: If the action space is not a tuple of discrete spaces.
        """
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        self.factor_sizes = [space.n for space in env.action_space.spaces]
        self.action_space = gym.spaces.Discrete(np.prod(self.factor_sizes))

    def action(self, action):
        """Overrides the default method of the superclass to translate a discrete action
        into a tuple of discrete actions.

        Args:
            action (int): The discrete action.

        Returns:
            list: A list of discrete actions.
        """
        unflattened_action = []
        self.factor_sizes[0] = self.factor_sizes[1] * self.factor_sizes[2]
        action, remainder = divmod(action, self.factor_sizes[0])
        unflattened_action.append(action)

        action_x, action_y = divmod(remainder, self.factor_sizes[2])
        unflattened_action.extend([action_x, action_y])

        return unflattened_action

    def validate_action(self, action):
        """Overrides the default method of the superclass to
        validate the action.

        Args:
            action (int): The action to be validated.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        return self.env.validate_action(*self.action(action))


class FlatteningActionMaskObservationWrapperSquare(gym.ObservationWrapper):
    """A Gym environment observation wrapper for the square
    environment.

    This wrapper flattens the action mask in the observation space.

    Attributes:
        observation_space (gym.spaces.Dict): The observation space.
    """

    def __init__(self, env):
        """Initializes the observation wrapper.

        Fllatens the action mask in the observation space.

        Args:
            env (gym.Env): The original Gym environment.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.utils.flatten_space(v) if k == "action_mask" else v
                for k, v in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation):
        """Overrides the default method of the superclass to
        flatten the "action_mask" space in the observation dictionary.

        Args:
            observation (dict): The original observation dictionary.

        Returns:
            dict: The modified observation dictionary with the "action_mask" space
                flattened.
        """
        return {
            k: gym.spaces.utils.flatten(self.observation_space[k], v)
            if k == "action_mask"
            else v
            for k, v in observation.items()
        }


class FlatteningActionWrapperSquare(gym.ActionWrapper):
    """A Gym environment action wrapper for the square
    environment.

    Action wrapper which flattens a tuple of discrete actions into
    a single discrete action for the square environment.

    Attributes:
        factor_sizes (list): The list of the sizes of the discrete action spaces.
        action_space (gym.spaces.Discrete): The discrete action space.
    """

    def __init__(self, env):
        """Initializes the action wrapper.

        Args:
            env (gym.Env): The original Gym environment.

        Raises:
            AssertionError: If the action space is not a tuple of discrete spaces.
        """
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        self.factor_sizes = [space.n for space in env.action_space.spaces]
        self.action_space = gym.spaces.Discrete(np.prod(self.factor_sizes))

    def action(self, action):
        """Overrides the default method of the superclass to translate a discrete action
        into a tuple of discrete actions.

        Args:
            action (int): The discrete action.

        Returns:
            list: A list of discrete actions.
        """
        unflattened_action = []

        action_x, action_y = divmod(action, self.factor_sizes[1])
        unflattened_action.extend([action_x, action_y])

        return unflattened_action

    def validate_action(self, action):
        """Overrides the default method of the superclass to
        validate the action.

        Args:
            action (int): The action to be validated.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        return self.env.validate_action(*self.action(action))
