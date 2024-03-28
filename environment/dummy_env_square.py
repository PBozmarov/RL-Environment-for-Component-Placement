"""
Dummy placement environment for square components.
"""

from typing import List, Tuple, Dict, Any
import gym  # type: ignore
import numpy as np


class DummyPlacementEnv(gym.Env):
    """A NumPy-based dummy environment for component placement.

    The environment is a grid of size height x width. The agent can place
    components on the grid, where the components are square n x n arrays. At
    each time step, the agent places a component on the grid by specifying the
    position of the top-left corner of the component on the grid, where the goal
    is to maximize the number of components placed on the grid.

    The observation space consists of the "grid" and "action_mask". The reward
    for the environment is 1 for each time step where a component is placed
    on the grid, and 0 otherwise. The environment is terminated when
    it is not possible to place any more components on the grid.

    Attributes:
        height (int): Height of the grid.
        width (int): Width of the grid.
        component_n (int): Size of square component.
        action_space (gym.spaces.Tuple): The action space which is a grid of size height x width.
        observation_space (gym.spaces.Dict): The observation space which is a dictionary with two keys: "grid" and "action_mask":
            - The grid is a NumPy array of shape (height, width) contining 1 for occupied cells and 0 for unoccupied cells.
            - The action mask is a NumPy array of shape (height, width) contining 1 for valid actions and 0 for invalid actions.

    Note:
        In this case observation space fully reflects the state of the environment.
    """

    def __init__(self, height: int, width: int, component_n: int) -> None:
        """Insantiates a new DummyPlacementEnv.

        Args:
            height (int): Height of the grid.
            width (int): Width of the grid.
            component_n (int): Size of square component.

        Raises:
            ValueError: If the grid size is less than or equal to 0 or if the component size is greater than the grid size.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.component_n = component_n

        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(height), gym.spaces.Discrete(width)]
        )
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0, high=1, shape=(height, width), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(height, width), dtype=np.float32
                ),
            }
        )

        if self.height < 0 or self.width < 0:
            raise ValueError("Grid size must be greater than 0.")
        if self.component_n > self.height or self.component_n > self.width:
            raise ValueError(
                "Component size must be less than or equal to the grid size."
            )

    def reset(
        self, verbose: bool = False, *args: Any, **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """Resets the environment.

        Args:
            verbose (bool): Whether to print the grid and action mask after the
                environment is reset.

        Returns:
            observation (dict): The initial observation.
        """
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.action_mask = np.ones((self.height, self.width), dtype=np.float32)
        self.components: List[Tuple[int, int]] = []
        self.actions: List[Tuple[int, int]] = []

        # Update the action mask for where the components cannot be placed
        # at the beginning of the episode
        if self.component_n > 1:
            cols_to_mask = [
                i
                for i in range(self.width - 1, self.width - self.component_n, -1)
                if i >= 0
            ]
            rows_to_mask = [
                i
                for i in range(self.height - 1, self.height - self.component_n, -1)
                if i >= 0
            ]
            self.action_mask[rows_to_mask, :] = 0
            self.action_mask[:, cols_to_mask] = 0

        # Print, grid, action mask if verbose
        if verbose:
            printed_string = "Grid:\n{}\n\nAction mask:\n{}\n\n"
            print(printed_string.format(self.grid, self.action_mask))

        obs = {"grid": self.grid.copy(), "action_mask": self.action_mask.copy()}
        return obs

    def step(
        self, action: Tuple[int, int], verbose: bool = False
    ) -> Tuple[dict, float, bool, dict]:
        """Steps the environment.

        Args:
            action (tuple): A tuple of (x,y) for the action to take.
            verbose (bool): Whether to print the grid and action mask after the
                action is taken.

        Returns:
            observation (dict): The observation after the action is taken.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is terminated.
            info (dict): Additional information.
        """
        x, y = action
        valid_action = self.validate_action(x, y)

        if not valid_action:
            return (
                {"grid": self.grid.copy(), "action_mask": self.action_mask.copy()},
                0.0,
                True,
                {},
            )
        # Place the component on the grid
        self.update_grid(x, y)
        self.update_action_mask(x, y)
        done = self.compute_if_done()
        reward = 1.0
        obs = {"grid": self.grid.copy(), "action_mask": self.action_mask.copy()}
        info: dict = {}

        # Print the grid, action mask, and reward if verbose
        if verbose:
            printed_string = "Grid:\n{}\n\nAction mask:\n{}\n\nReward: {}\n\n"
            print(printed_string.format(self.grid, self.action_mask, reward))
        return obs, reward, done, info

    def validate_action(self, x: int, y: int) -> bool:
        """Validates an action.

        The action is invalid if it would cause the component to go out of
        bounds or if any of cells that the component would occupy are already
        occupied. Note that the action mask both checks if the action is out
        of bounds and if any of the cells are already occupied.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.

        Returns:
            bool: Whether the action is valid.
        """
        try:
            return self.action_mask[x, y] == 1
        except IndexError:
            return False

    def update_grid(self, x: int, y: int) -> None:
        """Updates the grid with an action.

        The action is to place a component on the cell (x, y),
        where the tuple represents the top-left corner of the component.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.
        """
        self.grid[x : x + self.component_n, y : y + self.component_n] = 1

    def update_action_mask_horizontal(self, x: int, y: int) -> None:
        """Updates the action mask horizontally (left) with an action.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.
        """
        h_col_min, h_col_max = max(0, y - self.component_n + 1), y
        h_row_min, h_row_max = x, x + self.component_n
        self.action_mask[h_row_min:h_row_max, h_col_min:h_col_max] = 0

    def update_action_mask_vertical(self, x: int, y: int) -> None:
        """Updates the action mask vertically (upwards) with an action.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.
        """
        v_row_min, v_row_max = max(0, x - self.component_n + 1), x
        v_col_min, v_col_max = y, y + self.component_n
        self.action_mask[v_row_min:v_row_max, v_col_min:v_col_max] = 0

    def update_action_mask_diagonal(self, x: int, y: int) -> None:
        """Updates the action mask diagonally (upwards and left) with
           an action.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.
        """
        d_row_min, d_row_max = max(0, x - self.component_n + 1), x
        d_col_min, d_col_max = max(0, y - self.component_n + 1), y
        self.action_mask[d_row_min:d_row_max, d_col_min:d_col_max] = 0

    def update_action_mask(self, x: int, y: int) -> None:
        """Updates the action mask with an action.

        Args:
            x (int): x coordinate of the action.
            y (int): y coordinate of the action.
        """
        # Set the action mask to 0 for the cells that are occupied
        # by the component
        self.action_mask[x : x + self.component_n, y : y + self.component_n] = 0

        # Update the action mask for where the components would overlap
        if self.component_n > 1:
            # Updating horizontally
            if y != 0:
                self.update_action_mask_horizontal(x, y)

            # Updating vertically
            if x != 0:
                self.update_action_mask_vertical(x, y)

            # Updating diagonally
            if x != 0 and y != 0:
                self.update_action_mask_diagonal(x, y)

    def compute_if_done(self) -> bool:
        """Computes whether the episode is terminated.

        The episode is terminated when it is not possible to place any more
        components on the grid.

        Returns:
            bool: Whether the episode is terminated.
        """
        return bool(np.all(self.action_mask == 0))

    def __str__(self) -> str:
        """Returns a string representation of the grid."""
        return str(self.grid)
