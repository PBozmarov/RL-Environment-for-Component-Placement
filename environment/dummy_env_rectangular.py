"""
Dummy placement environment for rectangular components.
"""

import gym  # type: ignore
import numpy as np
from scipy import signal  # type: ignore
from typing import Tuple, List, Any


class Component(object):
    """A component object to be placed on the grid.

    Attributes:
        h (int): The height of the component.
        w (int): The width of the component.
        area (int): The area of the component.
        comp_id (int): The id of the component relative to other components.
        placed (bool): Whether the component has been placed on the grid.
        position (Tuple[float, float]): The position of the component on the grid.
    """

    def __init__(self, h: int, w: int, comp_id: int, placed: bool = False):
        """Instantiates a new Component object.

        Args:
            h (int): The height of the component.
            w (int): The width of the component.
            comp_id (int): The id of the component relative to other components.
            placed (bool): Whether the component has been placed on the grid.
        """
        self.h = h
        self.w = w
        self.area = h * w
        self.comp_id = comp_id
        self.placed = placed
        self.position = (-1.0, -1.0)

    def place_component(self, x: int, y: int):
        """Places the component on the grid at the given position.

        Args:
            x (int): The x coordinate of the position.
            y (int): The y coordinate of the position.
        """
        self.placed = True
        self.position = (x, y)

    def area_ratio(self, grid_area: int) -> float:
        """Returns the ratio of the component area to the grid area.

        Args:
            grid_area (int): The area of the grid.

        Returns:
            float: The ratio of the component area to the grid area.
        """
        return self.area / grid_area

    def calculate_feature(self, grid_area: int) -> np.ndarray:
        """Calculates a vector of features for the component.

        The vector of features is of the form [h, w, x, y, area_ratio], where
        h and w are the height and width of the component, x and y are the
        coordinates of the component on the grid, and area_ratio is the ratio
        of the component area to the grid area.

        Args:
            grid_area (int): The area of the grid.

        Returns:
            np.ndarray: The feature vector for the component.
        """
        component_x, component_y = self.position
        component_area_ratio = self.area_ratio(grid_area)
        feature = np.array(
            [self.h, self.w, component_x, component_y, component_area_ratio]
        )
        return feature

    def __str__(self) -> str:
        """Returns a string representation of the component.

        Returns:
            str: A string containing the Component's id, height, and width.
        """
        return f"Component {self.comp_id}: {self.h} x {self.w}"

    def __repr__(self) -> str:
        """Calls the __str__ method.

        Returns:
            str: A string containing the Component's id, height, and width.
        """
        return self.__str__()


class DummyPlacementEnv(gym.Env):
    """A NumPy-based dummy environment for component placement.
    The environment is a grid of size height x width. The agent can place
    components on the grid, where the components are rectangular arrays with
    dimensions between min_component_h x min_component_w. The goal is to place
    as many of the components on the grid as possible.

    At reset, a list of components is generated, where the dimensions of each
    component are randomly sampled from the range [min_component_h, max_component_h]
    and the number of components is randomly sampled from the range
    [min_num_components, max_num_components]. The agent takes actions
    by placing components, in the order that they are generated, on the grid by
    specifying the orientation and position of the top-left corner of the component
    on the grid.

    The observation space consists of the "grid", "action_mask", "all_components_feature",
    "component_mask", and "placement_mask". The reward is 1 for each component
    placed on the grid, and 0 otherwise. The environment is terminated when
    it is not possible to place any more components on the grid.

    Attributes:
        height (int): The height of the grid.
        width (int): The width of the grid.
        min_component_h (int): The minimum height of a component.
        min_component_w (int): The minimum width of a component.
        max_component_h (int): The maximum height of a component.
        max_component_w (int): The maximum width of a component.
        max_num_components (int): The maximum number of components.
        action_space (gym.spaces.Tuple): The space of all possible actions
            which are tuples of three integers (orientation, x, y), where orientation
            is 0 for placing the component in its original (a x b) orientation and 1
            for placing the component in its rotated (b x a) orientation, and x and y
            are the coordinates of the top-left corner of the component.
        observation_space (gym.spaces.Dict): The space of all possible observations which is
            a dictionary with the keys "grid", "action_mask", "all_components_feature",
            "component_mask", and "placement_mask".
        components (List[Component]): A list of the components in the environment.
        grid (np.ndarray): A NumPy array of shape (height, width) containing 1 for
            occupied cells and 0 for empty cells.
        action_mask (np.ndarray): A NumPy array of shape (2, height, width) containing 1
            for valid actions and 0 for invalid actions, where the first dimension
            corresponds to the two possible orientations of the component.
        all_components_feature (np.ndarray): A NumPy array of shape (max_num_components, 5)
            containing the features of all components. The features for each component are:
            component_h, component_w, component_x, component_y, and component_area_ratio.
        component_mask (np.ndarray): A NumPy array of shape (max_num_components, ) containing
            1 for components that are in the generated instance and 0 otherwise.
        placement_mask (np.ndarray): A NumPy array of shape (max_num_components, ) containing 1
            for components that have been placed on the grid and 0 otherwise.

    Note:
        In this case observation space fully reflects the state of the environment.
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_component_w: int,
        max_component_w: int,
        min_component_h: int,
        max_component_h: int,
        max_num_components: int,
        min_num_components: int,
    ):
        """Insantiates a new DummyPlacementEnv.

        Args:
            height (int): The height of the grid.
            width (int): The width of the grid.
            min_component_h (int): The minimum height of a component.
            min_component_w (int): The minimum width of a component.
            max_component_h (int): The maximum height of a component.
            max_component_w (int): The maximum width of a component.
            max_num_components (int): The maximum number of components.
            min_num_components (int): The minimum number of components.

        Raises:
            ValueError: - if the grid size is less than or equal to 0.
                        - if the component size is greater than the grid size.
                        - if the component size is less than or equal to 0.
                        - if the maximum number of components is  greater than the area of the grid
                            or less than or equal to 0.
        """
        # Initialize the environment
        self.height = height
        self.width = width
        self.area = height * width

        # Initialize component attributes
        self.min_component_w = min_component_w
        self.max_component_w = max_component_w
        self.min_component_h = min_component_h
        self.max_component_h = max_component_h
        self.max_num_components = max_num_components
        self.min_num_components = min_num_components
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(height),
                gym.spaces.Discrete(width),
            ]
        )

        # Initialize the observation space
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0, high=1, shape=(height, width), dtype=np.float64
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(2, height, width), dtype=np.float64
                ),
                "all_components_feature": gym.spaces.Box(
                    low=-1,
                    high=max(height, width),
                    shape=(max_num_components, 5),
                    dtype=np.float64,
                ),
                "component_mask": gym.spaces.Box(
                    low=0, high=1, shape=(max_num_components,), dtype=np.float64
                ),
                "placement_mask": gym.spaces.Box(
                    low=0, high=1, shape=(max_num_components,), dtype=np.float64
                ),
            }
        )

        # Initialize the environment state
        self.components: List[Component] = []
        self.grid: np.ndarray = np.zeros((height, width))
        self.action_mask: np.ndarray = np.zeros((2, height, width))
        self.placement_mask = np.zeros(self.max_num_components)
        self.component_mask = np.zeros(self.max_num_components)

        # Intialize all components feature which contains
        # (component_x, component_y, x_position_grid, y_position_grid,
        # area_ratio)
        self.all_components_feature: np.ndarray = np.zeros((self.max_num_components, 5))

        # Validate the environment parameters
        if self.height < 0 or self.width < 0:
            raise ValueError("Grid size must be greater than 0.")
        if self.max_component_w > self.height or self.max_component_h > self.width:
            raise ValueError(
                "Component size must be less than or equal to the grid size."
            )
        if self.min_component_w < 1 or self.min_component_h < 1:
            raise ValueError("Component size must be greater than 0.")
        if max_num_components < 1 or max_num_components > self.area:
            raise ValueError(
                "Number of components must be greater than 0"
                + "and less than or equal to the grid area."
            )

    def generate_instances(self):
        """Generates a list of components to be placed on the grid.

        Generates a list of components to be placed on the grid by
        randomly sampling the number of components and the size of
        each component.
        """
        no_components = np.random.randint(
            self.min_num_components, self.max_num_components + 1
        )
        components = []
        for i in range(no_components):
            component_h = np.random.randint(
                self.min_component_h, self.max_component_h + 1
            )
            component_w = np.random.randint(
                self.min_component_w, self.max_component_w + 1
            )
            components.append(Component(component_h, component_w, i))

        self.components = components

    def update_placement_mask(self, component: Component):
        """Updates the placement_mask array.

            Update the placement mask array to 1 for the
            given component to indicate that it has been placed.

        Args:
            component (Component): Component to be placed.
        """
        self.placement_mask[component.comp_id] = 1.0

    def update_component_mask(self, components: List[Component]):
        """Updates the component_mask array.

            Update the component mask array to 1 for the
            components that exist in the genrated instance and
            0 for the components that do not exist.

        Args:
            components (List[Component]): List of components.
        """
        self.component_mask = np.zeros(self.max_num_components)
        for component in components:
            self.component_mask[component.comp_id] = 1.0

    def update_all_components_feature(self, components: List[Component]):
        """Updates the all_components_feature given a list of components.
        Args:
            components (List[Component]): List of components.
        """
        for component in components:
            self.all_components_feature[
                component.comp_id, :
            ] = component.calculate_feature(self.area)

    def reset(self, verbose: bool = False, *args: Any, **kwargs: Any) -> dict:
        """Resets the environment.

            At each reset, a list of components is generated and the
            environment state is initialized. Following reset, the
            components are stepped through in the order that they were
            generated.

        Args:
            verbose (bool): Whether to print the grid and action mask after the
            environment is reset.

        Returns:
            state (dict): Dictionary containing the current environment state.
        """
        # Generate instances of components
        self.grid = np.zeros((self.height, self.width))
        self.generate_instances()

        # Initialize masks and all_components_features
        self.all_components_feature = np.zeros((self.max_num_components, 5))
        self.current_component = self.components[0]
        self.action_mask = self.compute_action_mask(self.current_component)

        # Update masks and features
        self.placement_mask = np.zeros(self.max_num_components)
        self.update_component_mask(self.components)
        self.update_all_components_feature(self.components)

        # Print grid and action mask if verbose
        if verbose:
            print_string = "Grid:\n{}\n\nAction mask:\n{}\n\n"
            print(print_string.format(self.grid, self.action_mask))

        state = {
            "grid": self.grid.copy(),
            "action_mask": self.action_mask.copy(),
            "all_components_feature": self.all_components_feature.copy(),
            "component_mask": self.component_mask.copy(),
            "placement_mask": self.placement_mask.copy(),
        }
        return state

    def step(
        self, action: Tuple[int, int, int], verbose: bool = False
    ) -> Tuple[dict, float, bool, dict]:
        """Steps the environment.

            Steps the environment with the given action using the
            current component.

        Args:
            action (Tuple[int, int, int]): The action to be taken,
                which is a tuple of (orientation, x, y).
                top left corner of the component on the grid.
            verbose (bool): Whether to print the grid and action mask after the
                environment is stepped through.

        Returns:
            observation (dict): The observation after the action is taken.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        orientation, x, y = action
        valid_action = self.validate_action(orientation, x, y)

        if valid_action:
            # Place the component on the grid
            self.update_grid(self.current_component, orientation, x, y)
            self.current_component.place_component(x, y)

            # Update the all components feature
            self.all_components_feature[
                self.current_component.comp_id, :
            ] = self.current_component.calculate_feature(self.area)

            # Update the placement mask
            self.update_placement_mask(self.current_component)

            # Set current component to next component
            if self.current_component.comp_id + 1 < len(self.components):
                self.current_component = self.components[
                    self.current_component.comp_id + 1
                ]
            else:
                self.current_component = Component(-1, -1, -1)

            # Compute action mask for next component
            if self.current_component.comp_id != -1:
                self.action_mask = self.compute_action_mask(self.current_component)
            else:
                self.action_mask = np.zeros((2, self.height, self.width))

            # Calculate the reward
            done = self.compute_if_done()
            reward = float(valid_action)
            info: dict = {}

            # Create observation
            obs = {
                "grid": self.grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "component_mask": self.component_mask.copy(),
                "placement_mask": self.placement_mask.copy(),
            }

            # Print the grid, action mask and reward if verbose
            if verbose:
                printed_string = "Grid:\n{}\n\nAction mask:\n{}\n\nReward: {}\n\n"
                print(printed_string.format(self.grid, self.action_mask, reward))
            return obs, reward, done, info

        else:
            obs = {
                "grid": self.grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "component_mask": self.component_mask.copy(),
                "placement_mask": self.placement_mask.copy(),
            }
            return obs, 0.0, True, {}

    def validate_action(self, orientation: int, x: int, y: int) -> bool:
        """Validates the given action.

            The action is invalid if it would cause the component to go out of
            bounds or if any of cells that the component would occupy are already
            occupied. Note that the action mask both checks if the action is out
            of bounds and if any of the cells are already occupied.

        Args:
            x (int): The x coordinate of the top left corner of the component.
            y (int): The y coordinate of the top left corner of the component.
            orientation (int): The orientation of the component (0 or 1),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is placed in its rotated
                (b x a) orientation.

        Returns:
            bool: Whether the action is valid.
        """
        try:
            return self.action_mask[orientation, x, y] == 1
        except IndexError:
            return False

    def update_grid(self, component: Component, orientation: int, x: int, y: int):
        """Updates the grid for the given component and action.

        Args:
            component (Component): The component to place on the grid
            x (int): The x coordinate of the top left corner of the component.
            y (int): The y coordinate of the top left corner of the component.
            orientation (int): The orientation of the component (0 or 1),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is placed in its rotated
                (b x a) orientation.

        Raises:
            Exception: If the orientation is not 0 or 1.
        """
        # Set the placement height and width based on the orientation
        if orientation == 0:
            placement_height = component.h
            placement_width = component.w
        elif orientation == 1:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")

        # Place the component on the grid
        self.grid[x : x + placement_height, y : y + placement_width] = 1

    def rows_cols_to_mask(
        self, component: Component, orientation: int
    ) -> Tuple[list, list]:
        """Gets the rows and cols to mask for a given orientation and component.

        Args:
                component (Component): The component to place on the grid
                orientation (int): The orientation of the component (0 or 1),
                    where 0 means the component is placed in its original (a x b)
                    orientation and 1 means the component is placed in its rotated
                    (b x a) orientation.

            Returns:
                rows_to_mask (list[int]): The rows to mask
                cols_to_mask (list[int]): The columns to mask

            Raises:
                Exception: If the orientation is not 0 or 1.
        """
        # Set the placement height and width based on the orientation
        if orientation == 0:
            placement_height = component.h
            placement_width = component.w
        elif orientation == 1:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")

        # Get the rows and cols to mask
        rows_to_mask = [
            i
            for i in range(self.height - 1, self.height - placement_height, -1)
            if i >= 0
        ]
        cols_to_mask = [
            i for i in range(self.width - 1, self.width - placement_width, -1) if i >= 0
        ]
        return rows_to_mask, cols_to_mask

    def compute_action_mask_orientation(
        self, component: Component, orientation: int
    ) -> np.ndarray:
        """Compute the action mask for the given component and orientation.

        Args:
            component (Component): The component to place on the grid
            orientation (int): The orientation of the component (0 or 1),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is placed in its rotated
                (b x a) orientation.

        Returns:
            action_mask_orientation (np.ndarray): The component's action mask
                for the given orientation.

        Raises:
            Exception: If the orientation is not 0 or 1.
        """
        # Assign placement height and width based on orientation
        if orientation == 0:
            placement_height = component.h
            placement_width = component.w
        elif orientation == 1:
            placement_height = component.w
            placement_width = component.h
        else:
            raise Exception("Invalid orientation.")

        # Initialise the action mask
        action_mask_orientation = np.ones((self.height, self.width))

        # Mask rows and columns near boundary.
        rows_to_mask, cols_to_mask = self.rows_cols_to_mask(component, orientation)
        action_mask_orientation[rows_to_mask, :] = 0
        action_mask_orientation[:, cols_to_mask] = 0

        mask = np.ones((placement_height, placement_width))
        action_mask_orientation[
            : self.height - placement_height + 1, : self.width - placement_width + 1
        ] = np.where(signal.convolve2d(self.grid, mask, mode="valid") == 0, 1, 0)
        return action_mask_orientation

    def compute_action_mask(self, component: Component):
        """Compute the action mask for the given component.

        Args:
            component (Component): The component to place on the grid.

        Returns:
            action_mask (np.ndarray): The action mask for the given component.
        """
        # Initialise the action mask
        action_mask = np.zeros((2, self.height, self.width))

        # Compute the action mask for each orientation
        action_mask[0] = self.compute_action_mask_orientation(component, 0)
        action_mask[1] = self.compute_action_mask_orientation(component, 1)
        return action_mask

    def compute_if_done(self) -> bool:
        """Computes whether the episode is terminated.

        The episode is terminated when it is not possible to place the
        next component on the grid or all components have been placed.

        Returns:
            bool: Whether the episode is terminated
        """
        # Check if there are no more components to place
        if self.current_component.comp_id != -1:
            # Check if there are no more actions to take
            return bool(np.all(self.action_mask == 0))
        return True

    def __str__(self) -> str:
        """Returns a string representation of the grid."""
        return str(self.grid)
