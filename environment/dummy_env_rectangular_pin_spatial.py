"""
Dummy placement environment for rectangular components with pins, where the components, pins and nets
are represented spatially in the observation space.
"""

import gym  # type: ignore
import heapq
import numpy as np
import random
from scipy import signal  # type: ignore
from typing import Dict, Tuple, Set, List, Any


class Pin(object):
    """A pin to be added to a component.

    Attributes:
        relative_x (int): The x position of the pin relative to the top-left
            corner of the component.
        relative_y (int): The y position of the pin relative to the top-left
            corner of the component.
        absolute_x (int): The x position of the pin relative to the top-left
            corner of the grid.
        absolute_y (int): The y position of the pin relative to the top-left
            corner of the grid.
        pin_id (int): The id of the pin relative to the component.
        component_id (int): The id of the component the pin belongs to.
        net_id (int): The id of the net the pin is assigned to.
    """

    def __init__(
        self,
        relative_x: int,
        relative_y: int,
        pin_id: int,
        component_id: int,
        net_id: int,
    ):
        """Initializes the pin.
        Args:
            relative_x (int): The x position of the pin relative to the top-left
                corner of the component.
            relative_y (int): The y position of the pin relative to the top-left
                corner of the component.
            pin_id (int): The id of the pin relative to the component.
            component_id (int): The id of the component the pin belongs to.
            net_id (int): The id of the net the pin is assigned to.
        """
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.absolute_x = -1
        self.absolute_y = -1
        self.pin_id = pin_id
        self.component_id = component_id
        self.net_id = net_id

    def update_pin_position(self, component_x: int, component_y: int):
        """Updates the absolute position of the pin based on the position of the
        top-left corner of the component that the pin is placed on.

        Args:
            component_x (int): The x position of the top-left corner of the
                            component that the pin belongs to.
            component_y (int): The y position of the top-left corner of the
                            component that the pin belongs to.
        """
        self.absolute_x = component_x + self.relative_x
        self.absolute_y = component_y + self.relative_y

    def calculate_feature(
        self, component_x: int, component_y: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the feature vector for the pin.

        Args:
            component_x (int): The x position of the top-left corner of the
                            component that the pin belongs to.
            component_y (int): The y position of the top-left corner of the
                            component that the pin belongs to.

        Returns:
            Tuple[np.ndarray, np.ndarray] : The numerical feature vector for the pin, the categorical
                        feature vector for the pin.
        """
        if not (component_x == -1 or component_y == -1):
            self.update_pin_position(component_x, component_y)

        num_features = np.array(
            [
                self.relative_x,
                self.relative_y,
                self.absolute_x,
                self.absolute_y,
            ]
        )

        cat_features = np.array(
            [
                self.net_id,
                self.component_id,
            ]
        )

        return num_features, cat_features

    def __str__(self) -> str:
        """Returns a string representation of the pin."""
        return f"Pin at ({self.absolute_x, self.absolute_y}) for component {self.component_id} assigned to net {self.net_id}"

    def __repr__(self) -> str:
        """Returns a string representation of the pin."""
        return self.__str__()


class Component(object):
    """A component object to be placed on the grid.

    Attributes:
        h (int): The height of the component.
        w (int): The width of the component.
        area (int): The area of the component.
        comp_id (int): The id of the component relative to other components.
        placed (bool): Whether the component has been placed on the grid.
        position (Tuple[float, float]): The position of the component on the grid.
        pins (List[Pin]): The pins to be added to the component.
    """

    def __init__(
        self, h: int, w: int, comp_id: int, pins: List[Pin], placed: bool = False
    ):
        """Initializes the component.

        Args:
            h (int): The height of the component.
            w (int): The width of the component.
            comp_id (int): The id of the component relative to other components.
            pins (List[Pin]): The pins to be added to the component.
            placed (bool, optional): Whether the component has been placed on the grid.
                Defaults to False.
        """
        self.h = h
        self.w = w
        self.area = h * w
        self.comp_id = comp_id
        self.placed = placed
        self.position = (-1, -1)
        self.pins = pins

    def place_component(self, orientation: int, x: int, y: int):
        """Places the component on the grid.

            Places the component on the grid by updating its position and
            by updating the relative and absolute positions of the pins
            based on the orientation of the component.

        Args:
            orientation (int): The orientation of the component. 0 is the original
                orientation, 1 is 90 degrees clockwise, 2 is 180 degrees
                clockwise, and 3 is 270 degrees clockwise.
            x (int): The x position of the top-left corner of the component.
            y (int): The y position of the top-left corner of the component.
        """
        self.placed = True
        self.position = (x, y)

        # Update the positions of pins
        if orientation == 0:  # original orientation
            for pin in self.pins:
                pin.update_pin_position(x, y)
        elif orientation == 1:  # 90 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    pin.relative_y,
                    self.h - pin.relative_x - 1,
                )
                pin.update_pin_position(x, y)
        elif orientation == 2:  # 180 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    self.h - pin.relative_x - 1,
                    self.w - pin.relative_y - 1,
                )
                pin.update_pin_position(x, y)
        elif orientation == 3:  # 270 degrees clockwise
            for pin in self.pins:
                pin.relative_x, pin.relative_y = (
                    self.w - pin.relative_y - 1,
                    pin.relative_x,
                )
                pin.update_pin_position(x, y)

    def area_ratio(self, grid_area: int) -> float:
        """Returns the ratio of the component area to the grid area.

        Args:
            grid_area (int): Total area of the grid.

        Returns:
            float: The ratio of the component area to the grid area.
        """
        return self.area / grid_area

    def calculate_feature(
        self, grid_area: int, max_num_pins_per_component: int
    ) -> np.ndarray:
        """Calculates the feature vector for the component.

            The feature vector contains:
                - the height of the component
                - the width of the component
                - the x position of the component
                - the y position of the component
                - the ratio of the component area to the grid area

        Args:
            grid_area (int): The area of the grid.
            max_num_pins_per_component (int): The maximum number of pins per component.

        Returns:
            np.ndarray: The feature vector for the component.
        """
        component_x, component_y = self.position
        component_area_ratio = self.area_ratio(grid_area)

        component_feature = np.array(
            [self.h, self.w, component_x, component_y, component_area_ratio],
        )

        # create an array of size max_num_pins_per_component with all values being -1
        all_pin_ids = np.full(max_num_pins_per_component, -1)
        # get a array of pin ids from the pins
        pin_ids = np.array([pin.pin_id for pin in self.pins])
        # replace the first len(pin_ids) values in all_pin_ids with the pin ids
        all_pin_ids[: len(pin_ids)] = pin_ids

        # extend the feature vector with the pin ids
        component_feature = np.append(component_feature, all_pin_ids)

        return component_feature

    def __str__(self) -> str:
        """Returns a string representation of the component."""
        return f"Component {self.comp_id}: {self.h} x {self.w}"

    def __repr__(self) -> str:
        """Returns a string representation of the component."""
        return self.__str__()


def sample_truncated_multinomial(n, m, p, k):
    """Sample from a multinomial distribution with a constraint on the maximum
    number of a sample for each bin.

    Args:
        n (int): The number of bins.
        m (int): The number of independent trials.
        p (array-like): The probabilities of the n bins.
        k (int): The maximum number of a sample for each bin.

    Returns:
        A list of integers representing the number of samples from each bin.

    Raises:
        ValueError: If k < 1 or k > m or if the sum of probabilities is not equal to 1.
    """
    if k < 1 or k > m:
        raise ValueError("Invalid value of k.")

    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities should sum to 1.")

    # Initialize the result with zeros
    result = np.zeros(n, dtype=int)

    # Loop through the number of independent trials
    for _ in range(m):
        # Calculate the truncated probabilities
        truncated_p = p * (result < k)
        truncated_p /= np.sum(truncated_p)

        # Sample from the truncated multinomial distribution
        sample = np.random.multinomial(1, truncated_p)

        # Update the result
        result += sample

    return result.tolist()


class DummyPlacementEnv(gym.Env):
    """A NumPy-based dummy environment for component placement.

    The environment is a grid of size height x width. The agent can place
    components on the grid, where the components are rectangular arrays with
    dimensions between min_component_h x min_component_w and have pins on them.
    The goal is to place the components on the grid such that routing can be done
    as efficiently as possible.

    At reset, a list of components is generated, where the dimensions of each
    component is in the range [min_component_h, max_component_h] and the number
    of components is in the [min_num_components, max_num_components]. The generated
    components also have pins on them which belong to "nets" specifying which pins
    are connected to each other. The agent takes actions by placing components, in
    in the order that they are generated, on the grid by specifying the orientation
    and position of the top-left corner of the component on the grid.

    The observation space consists of the "grid", "pin_grid", "component_grid",
     "action_mask", "all_components_feature", "all_pins_num_feature",
     "all_pins_cat_feature", "placement_mask".
    The reward used is sparse and incentivizes the agent to place components on the
    grid in such a way that routing can be done as efficiently as possible:

        - If the action taken is valid:
            - The reward is 0 if the component is not the
                last component to be placed on the grid.
            - If the component is the last component to be placed on the grid,
                the reward is calculated by first routing the nets in a way which
                approximates the optimal routing. The reward is then calculated
                by finding the number of intersections and the total wirelength
                of the routing and using these as a penalty.

        - If the action taken is invalid, then a lower bound for the
            worst possible reward is returned. This is to encourage the agent
            to place all components on the grid.

    The environment is terminated when either all the components have been placed
    or the action taken is invalid.

    Attributes:
        height (int): The height of the grid.
        width (int): The width of the grid.
        area (int): The area of the grid.
        net_distribution (int): An integer between 0 and 9 which determines
            the distribution of pins across nets. A higher value means that
            the pins are more evenly distributed across nets.
        pin_spread (int): An integer between 0 and 9 which determines the
            spread of pins on the components. A higher value means that the
            pins are more evenly distributed across the components.
        min_component_h (int): The minimum height of a component.
        max_component_h (int): The maximum height of a component.
        min_component_w (int): The minimum width of a component.
        max_component_w (int): The maximum width of a component.
        min_num_components (int): The minimum number of components.
        max_num_components (int): The maximum number of components.
        min_num_nets (int): The minimum number of nets.
        max_num_nets (int): The maximum number of nets.
        min_num_pins_per_net (int): The minimum number of pins per net.
        max_num_pins_per_net (int): The maximum number of pins per net.
        max_num_pins_per_component (int): The maximum number of pins per component.
        action_space (gym.spaces.Tuple): The space of all possible actions
            which are tuples of three integers (orientation, x, y), where orientation
            is 0 for placing the component in its original orientation and 1, 2, 3
            for placing the component rotated 90, 180, 270 degrees respectively, and
            x and y are the coordinates of the top-left corner of the component.
        pins (List[Pin]): A list of all the pins in the environment.
        reward_type (str): the type of routing to use in the reward function.
            Must be one of "beam", "centroid", "both".
        reward_beam_width (int): The beam width to use in the beam routing.
        weight_wirelength (float): The weight to use for the wirelength penalty.
        weight_num_intersections (float): The weight to use for the number of
            intersections penalty.
        max_num_intersections (int): The maximum number of intersections possible to
            use for the number of intersections penalty in the worst case scenario.
        max_wirelength (int): The maximum wirelength possible to use for the wirelength
            penalty in the worst case scenario.
        observation_space (gym.spaces.Dict): The space of all possible observations
            which is a dictionary with keys: "grid", "pin_grid", "component_grid",
            "action_mask", "all_components_feature", "all_pins_num_feature",
             "all_pins_cat_feature", "placement_mask".
        components (List[Component]): A list of the components in the environment.
        grid (np.ndarray): a NumPy array of shape (height, width) containing 1 for
            occupied cells and 0 for empty cells.
        action_mask (np.ndarray): A NumPy array of shape (2, height, width) containing 1
            for valid actions and 0 for invalid actions, where the first dimension
            corresponds to the two possible orientations of the component.
        pin_grid (np.ndarray): A NumPy array of shape (height, width, num_nets + 1). The
            array contains the current state of the pin grid. Each cell contains an integer
            0 or 1, where 0 means the cell is empty and 1 means the cell is occupied.
        component_grid (np.ndarray): A NumPy array of shape (max_num_components, height, width, num_nets + 1).
            The array contains a 3d representation of the components and distribution of pins on
            the components across differnet nets.
        placement_mask (np.ndarray): A NumPy array of shape (max_num_components, ) containing 0
            for components that don't exist in the instance, 1 for components that are
            to be placed, 2 for a component that has been placed and 3 for the current
            component being placed.
        all_components_feature (np.ndarray): A NumPy array of shape (max_num_components, 5)
            containing the features of all components. The features for each component are:
            component_h, component_w, component_x, component_y, and component_area_ratio.
        all_pins_num_feature (np.ndarray): A NumPy array of shape (max_num_components, max_num_pins_per_component, 4)
            containing the numerical features of all pins. The features for each pin are:
            relative_x, relative_y, absolute_x, absolute_y, net_id, component_id.
        all_pins_cat_feature (np.ndarray): A NumPy array of shape (max_num_components, max_num_pins_per_component, 1)
            containing the categorical features of all pins. The feature for each pin is the pin_id.
    """

    def __init__(  # noqa: max-complexity: 15
        self,
        height: int,
        width: int,
        net_distribution: int,
        pin_spread: int,
        min_component_w: int,
        max_component_w: int,
        min_component_h: int,
        max_component_h: int,
        max_num_components: int,
        min_num_components: int,
        min_num_nets: int,
        max_num_nets: int,
        max_num_pins_per_net: int,
        min_num_pins_per_net: int = 2,
        reward_type: str = "both",
        reward_beam_width: int = 2,
        weight_wirelength: float = 0.5,
        weight_num_intersections: float = 0.5,
    ):
        """Instantiates a new DummyPlacementEnv.

        Args:
            height (int): The height of the grid.
            width (int): The width of the grid.
            net_distribution (int): The distribution of the number of nets.
            pin_spread (int): The spread of the number of pins per net.
            min_component_w (int): The minimum width of a component.
            max_component_w (int): The maximum width of a component.
            min_component_h (int): The minimum height of a component.
            max_component_h (int): The maximum height of a component.
            max_num_components (int): The maximum number of components.
            min_num_components (int): The minimum number of components.
            min_num_nets (int): The minimum number of nets.
            max_num_nets (int): The maximum number of nets.
            max_num_pins_per_net (int): The maximum number of pins per net.
            min_num_pins_per_net (int, optional): The minimum number of pins per net.
                Defaults to 2.
            reward_type (str, optional): The type of routing to use in the reward function.
                Must be one of "beam", "centroid", "both". Defaults to "both".
            reward_beam_width (int, optional): The beam width to use in the beam routing.
                Defaults to 2.
            weight_wirelength (float, optional): The weight to use for the wirelength penalty.
                Defaults to 0.5.
            weight_num_intersections (float, optional): The weight to use for the number of
                intersections penalty. Defaults to 0.5.
        """
        # Initialize the environment
        self.height = height
        self.width = width
        self.area = height * width

        # Clip comlexity to [0, 9]
        self.net_distribution = max(0, min(9, net_distribution))
        self.pin_spread = max(0, min(9, pin_spread))

        # Initialize component attributes
        self.min_component_w = min_component_w
        self.max_component_w = max_component_w
        self.min_component_h = min_component_h
        self.max_component_h = max_component_h
        self.max_num_components = max_num_components
        self.min_num_components = min_num_components
        self.min_num_nets = min_num_nets
        self.max_num_nets = max_num_nets
        self.max_num_pins_per_net = max_num_pins_per_net
        self.min_num_pins_per_net = min_num_pins_per_net
        self.max_num_pins_per_component = self.max_component_h * self.max_component_w
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(4),
                gym.spaces.Discrete(height),
                gym.spaces.Discrete(width),
            ]
        )
        self.pins: List[Pin] = []
        self.reward_type = reward_type
        self.reward_beam_width = reward_beam_width
        self.weight_wirelength = weight_wirelength
        self.weight_num_intersections = weight_num_intersections
        self.max_num_intersections = self.calculate_upper_bound_intersections()
        self.max_wirelength = self.calculate_upper_bound_wirelength()
        self.component_grid = np.zeros(
            (
                self.max_num_components,
                self.max_component_h,
                self.max_component_w,
                self.max_num_nets + 1,
            )
        )

        # Initialize the observation space
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0, high=1, shape=(height, width), dtype=np.float64
                ),
                "pin_grid": gym.spaces.Box(
                    low=0,
                    high=100,
                    shape=(height, width, self.max_num_nets + 1),
                    dtype=np.float64,
                ),
                "component_grid": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        self.max_num_components,
                        self.max_component_h,
                        self.max_component_w,
                        self.max_num_nets + 1,
                    ),
                    dtype=np.float64,
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(4, height, width), dtype=np.float64
                ),
                "all_components_feature": gym.spaces.Box(
                    low=-1,
                    high=max(
                        self.max_num_pins_per_net * self.max_num_nets,
                        self.height * self.width,
                    ),
                    shape=(
                        self.max_num_components,
                        5 + self.max_num_pins_per_component,
                    ),
                    dtype=np.float64,
                ),
                "all_pins_num_feature": gym.spaces.Box(
                    low=-1,
                    high=max(height, width),
                    shape=(
                        self.max_num_components * self.max_num_pins_per_component + 1,
                        4,
                    ),
                    dtype=np.float64,
                ),
                "all_pins_cat_feature": gym.spaces.Box(
                    low=-1,
                    high=max(self.max_num_components, self.max_num_nets),
                    shape=(
                        self.max_num_components * self.max_num_pins_per_component + 1,
                        2,
                    ),
                    dtype=np.int32,
                ),
                "placement_mask": gym.spaces.Box(
                    low=0, high=3, shape=(max_num_components,), dtype=np.float64
                ),
            }
        )

        # Initialize the environment state
        self.components: List[Component] = []
        self.draw_components()
        self.grid: np.ndarray = np.zeros((height, width))
        self.pin_grid: np.ndarray = np.zeros((height, width, self.max_num_nets + 1))
        self.actions: List[Tuple[int, int, int]] = []
        self.action_mask: np.ndarray = np.zeros((4, height, width))
        self.placement_mask = np.ones(self.max_num_components)

        # Intialize all components feature which contains
        # (component_h, component_w, component_position_h, component_position_w,
        # area_ratio, list of pin_id's)
        # Initialize all pins feature which contains
        # (relative_x, relative_y, absolute_x, absolute_y, net_id, component_id)
        self.all_components_feature: np.ndarray = np.zeros(
            (self.max_num_components, 5 + self.max_num_pins_per_component),
            dtype=np.float64,
        )
        self.all_pins_num_feature: np.ndarray = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component + 1, 4)
        )
        self.all_pins_cat_feature: np.ndarray = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component + 1, 2)
        )
        self.all_pins_cat_feature[-1, :] = np.array([-1, -1])

        # Initialize the current component id
        self.current_component_id = -1

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
        if self.reward_type not in ["beam", "centroid", "both"]:
            raise ValueError("Invalid type of routing for the reward")
        if (
            self.reward_beam_width < 2
            or self.reward_beam_width > self.max_num_pins_per_net
        ):
            raise ValueError(
                "The beam width must be an integer greater than 2 and at most the maximum number of pins per net."
            )
        if not isinstance(self.reward_beam_width, int):
            raise ValueError("The beam width must be an integer.")
        if not isinstance(self.weight_wirelength, float):
            raise ValueError("The weight of wirelength must be a float.")
        if self.weight_wirelength < 0:
            raise ValueError("The weight of wirelength must be greater than 0.")

    def lowest_num_intersections(
        self, routes: List[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]]
    ) -> Tuple[int, int]:
        """Finds the lowest number of intersections from a list of routes.

        Args:
            routes (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): A list of
                routes where each route is a list of list of tuples; each list
                represents the routing of a net and each tuple represents a line
                segment (connection) defined by two points (pins) on the net.

        Returns:
            Tuple[int, int]: A tuple of the lowest number of intersections and the
                index of the route which achieves this.
        """
        num_intersections = []
        for route in routes:
            num_intersections.append(self.find_num_intersection(route))
        return min(num_intersections), num_intersections.index(min(num_intersections))

    def find_num_intersection(  # noqa: max-complexity: 15
        self, route: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    ) -> int:
        """Finds the number of intersections for a given route.

        Args:
            route (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): A list
                of list of tuples; each list represents the routing of a net and
                each tuple represents a line segment (connection) defined by two
                points (pins) on the net.

        Returns:
            int: Number of intersections.
        """
        num_intersections = 0
        for net in range(len(route)):
            for other_net in range(net + 1, len(route)):
                for line_segment_1 in route[net]:
                    for line_segment_2 in route[other_net]:
                        if self.is_intersect(line_segment_1, line_segment_2):
                            num_intersections += 1

        return num_intersections

    def is_intersect(
        self,
        line1: Tuple[Tuple[int, int], Tuple[int, int]],
        line2: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> bool:
        """Checks if two line segments intersect.

        Args:
            line1 (Tuple[Tuple[int, int], Tuple[int, int]]): A line segment defined
            by two points.
            line2 (Tuple[Tuple[int, int], Tuple[int, int]]): A line segment defined
            by two points.

        Returns:
            bool: True if the line segments intersect, False otherwise.
        """
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        if (
            line1[0] == line2[0]
            or line1[0] == line2[1]
            or line1[1] == line2[0]
            or line1[1] == line2[1]
        ):
            return True

        # Calculate the determinant
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # If the determinant is zero, there is no intersection
        if det == 0:
            return False

        # # Calculate the x and y coordinates of the intersection point
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        # Check if the intersection point lies on both line segments
        if (
            min(x1, x2) <= x <= max(x1, x2)
            and min(x3, x4) <= x <= max(x3, x4)
            and min(y1, y2) <= y <= max(y1, y2)
            and min(y3, y4) <= y <= max(y3, y4)
        ):
            return True
        else:
            return False

    def find_wirelength(
        self, route: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    ) -> float:
        """Finds the total wirelength of a route.

        Args:
            route (List[Tuple[Tuple[int, int], Tuple[int, int]]]): A list of list of
                tuples; each list represents the routing of a net and each tuple
                represents a line segment (connection) defined by two points (pins)
                on the net.

        Returns:
            float: Total wirelength.
        """
        wirelength = 0.0
        for net in route:
            for line in net:
                wirelength += self.euclidean_distance(line[0], line[1])
        return wirelength

    def calculate_upper_bound_wirelength(self) -> float:
        r"""Finds the upper bound on the total wirelength.

        To find the upper bound on the total wirelength for a given route,
        we assume that the worst case scenario is when the pins for each
        net are evenly placed in opposite corners of the grid, with pins
        being able to be superimposed on top of each other and with the routing
        being done using the "centroid" method.

        Assuming that we have a maximum of n nets and each net has a maximum
        of m pins, the upper bound on the total wirelength is given by:

        .. math::
            \frac{1}{2} \sqrt{h^2 + w^2} m*n

        Returns:
            float: Upper bound on wirelength.
        """
        distance = self.euclidean_distance((0, 0), (self.height, self.width))
        total_distance = (
            0.5 * distance * (self.max_num_nets * self.max_num_pins_per_net)
        )
        return total_distance / (self.height + self.width)

    def calculate_upper_bound_intersections(self) -> float:
        r"""Finds the upper bound on the number of intersections.

        To find the upper bound, we assume that the worst case scenario is
        when we are using the centroid routing method and each
        net has the same centroid.

        Suppose we have max_num_pins_per_net is :math:`n`, and max_num_nets is
        :math:`m`. For the first net, each (pin, centroid) pair will intersect
        with each (pin, centroid) pair for every other net, giving :math:`n^2`
        intersections for the first net. Overall, there will be:

        .. math::
            (m-1) * n^2

        intersections for the first net. For each subsequent net, we
        repeat the same method, except that we have one less net to
        intersect with. Therefore, the total number of intersections for
        the :math:`i`th net is:

        .. math::
            (m-i) * n^2

        This gives the following expression for the upper bound on the
        number of intersections:

        .. math::
            \sum_{i=1}^m n^2(m-i)

        which simplifies to:

        .. math::
            \frac{1}{2} n^2 m (m-1)

        Returns:
            float: upper bound on number of intersections
        """
        max_num_intersections = (
            0.5
            * (self.max_num_pins_per_net**2)
            * self.max_num_nets
            * (self.max_num_nets - 1)
        )
        return max_num_intersections

    def find_reward(self) -> float:
        r"""Finds the reward for an episode.

        As the reward used is sparse, the reward is only calculated at the
        end of an episode. The reward is calculated by first routing the nets according
        to the method specified by the reward_type attribute. Then, the reward is
        calculated as the weighted negative sum of the normalized wirelength and the
        normalized number of intersections for the routing:

        .. math::
            -(\lambda * \text{wirelength} + \mu * \text{number of intersections})

        The wirelength is normalized by :math:`\text{height} + \text{width}` and
        the weight is given by self.weight_wirelength. The number of intersections is normalized
        by considering the average number of pins for a randomly generated instance:

            - First, the average number of pins is calculated by multiplying
                average component size with average number of components.
            - Second, the average number of pins is calculated by
                multiplying the average number of pins with the average number of nets.
            - The minimum of these is used as the normalizing factor.

        To self.reward_type attribute can be one of:

            - "beam": uses beam search to find the best route with self.reward_beam_width
                as the beam width and the starting point as the point returned by the
                pin_outlier method
            - "centroid": uses centroid routing to find the best route. The centroid
                method finds the centroid for each net and connects all the pins in that
                net to the centroid.
            - "both": uses both beam search and centroid routing and chooses the routing
                with the least number of (unnormalized) intersections.

        For the scenario when not all the components in the generated instance have been
        placed in the episode, the reward is calculating by using an upper bound on
        the wirelength and the number of intersections. The idea behind this is to
        return a reward that is lower than any possible reward and therefore
        encourage the agent to place all the components on the grid.

        Returns:
            float: Reward for the current episode.

        Raises:
            ValueError: If beam search is used and the beam width is not a positive.
        """
        placed_all = self.current_component.comp_id == -1
        wirelength_normalizing_factor = self.height + self.width
        avg_total_num_pins_comp_size = (
            np.mean([self.min_component_h, self.max_component_h])
            * np.mean([self.min_component_w, self.max_component_w])
            * np.mean([self.min_num_components, self.max_num_components])
        )
        avg_total_num_pins_net = np.mean(
            [self.min_num_pins_per_net, self.max_num_pins_per_net]
        ) * np.mean([self.min_num_nets, self.max_num_nets])
        num_intersections_normalizing_factor = np.min(
            [avg_total_num_pins_comp_size, avg_total_num_pins_net]
        )

        # Reward for when not all components have been placed
        if not placed_all:
            reward = -self.weight_wirelength * (
                self.max_wirelength / wirelength_normalizing_factor
            ) - self.weight_num_intersections * (
                self.max_num_intersections / num_intersections_normalizing_factor
            )

            # Update the different reward components for callbacks
            self.reward_intersection = self.max_num_intersections
            self.reward_wirelength = self.max_wirelength
            return reward

        # Calculate reward for when all components have been placed
        if self.reward_type == "beam":
            if self.reward_beam_width == 0:
                raise ValueError("Beam search is used but beam_k is 0.")
            else:
                # Find the routes using beam search
                route = self.route_pins_beam_search(self.reward_beam_width)
                num_intersections = self.find_num_intersection(route)

                # Find the wirelength and reward
                wirelength = self.find_wirelength(route) / wirelength_normalizing_factor
                num_intersections = (
                    num_intersections / num_intersections_normalizing_factor
                )
                reward = -1 * (
                    self.weight_wirelength * wirelength
                    + self.weight_num_intersections * num_intersections
                )

                # Update the different reward components for callbacks
                self.reward_intersection = num_intersections
                self.reward_wirelength = wirelength

        elif self.reward_type == "centroid":
            # Find the routes using centroid routing
            route = self.route_pins_centroid()
            num_intersections = self.find_num_intersection(route)

            # Find the wirelength and reward
            wirelength = self.find_wirelength(route) / wirelength_normalizing_factor
            num_intersections = num_intersections / num_intersections_normalizing_factor
            reward = -1 * (
                self.weight_wirelength * wirelength
                + self.weight_num_intersections * num_intersections
            )

            # Update the different reward components for callbacks
            self.reward_intersection = num_intersections
            self.reward_wirelength = wirelength

        elif self.reward_type == "both":
            # Find the routes using beam search and centroid routing
            route_beam = self.route_pins_beam_search(self.reward_beam_width)
            route_centroid = self.route_pins_centroid()
            routes = [route_beam, route_centroid]

            # Find the min_num_intersections and wirelength
            min_num_intersections, index = self.lowest_num_intersections(routes)
            wirelength = (
                self.find_wirelength(routes[index]) / wirelength_normalizing_factor
            )
            min_num_intersections = (
                min_num_intersections / num_intersections_normalizing_factor
            )

            # Update the different reward components for callbacks
            self.reward_intersection = min_num_intersections
            self.reward_wirelength = wirelength

            # Find the reward depending on whether the episode is complete
            reward = -1 * (
                self.weight_wirelength * wirelength
                + self.weight_num_intersections * min_num_intersections
            )
        return reward

    def generate_pins(self):
        """Generates a list of pins to be placed on the grid."""
        self.pins = []
        for i in range(self.total_num_pins):
            self.pins.append(Pin(-1, -1, i, -1, -1))

    def generate_components(self):
        """Generates a list of components to be placed on the grid.

        Generates components by sampling from the minimum and maximum
        height and width of components.
        """
        components = []
        self.sample_num_components()
        for i in range(self.num_components):
            component_h = np.random.randint(
                self.min_component_h, self.max_component_h + 1
            )
            component_w = np.random.randint(
                self.min_component_w, self.max_component_w + 1
            )
            pins = []
            components.append(Component(component_h, component_w, i, pins))

        self.components = components
        self.total_area_covered_by_all_components = 0
        for component in self.components:
            self.total_area_covered_by_all_components += component.area

    def generate_instances(self):
        """Generates an instance of components to be placed on the grid.

        Generates an instance by (in order):

            - Generating a list of components to be placed on the grid using
                the generate_components method.
            - Sampling the number of nets using sample_num_nets().
            - Sampling the total number of pins using the sample_total_num_pins
                method.
            - Generating a list of pins to be placed on the grid using the
                generate_pins method
            - Allocating pins to nets using the allocate_pins_to_nets method.
            - Allocating pins to components using the allocate_pins_to_components
                method.
        """
        self.generate_components()
        self.sample_num_nets()
        self.sample_total_num_pins()
        self.generate_pins()
        self.allocate_pins_to_nets()
        self.allocate_pins_to_components()

        components_pins = {component.comp_id: [] for component in self.components}
        for pin in self.pins:
            components_pins[pin.component_id].append(pin)

        for component in self.components:
            component.pins = components_pins[component.comp_id]
            self.place_pins_on_component(component)

    def sample_num_components(self):
        """Samples the number of components to be placed on the grid."""
        self.num_components = np.random.randint(
            self.min_num_components, self.max_num_components + 1
        )

    def sample_num_nets(self):
        """Samples the number of nets to be placed on the grid."""
        self.num_nets = np.random.randint(self.min_num_nets, self.max_num_nets + 1)
        self.num_nets = min(
            self.num_nets, int(self.total_area_covered_by_all_components / 2)
        )

    def sample_total_num_pins(self):
        """Samples the total number of pins to be placed on the grid.

        Note:
            If the sampled total number of pins is greater than the total area
            covered by all components, the total number of pins is set to the
            total area covered by all components.
        """
        total_num_pins = np.random.randint(
            self.min_num_pins_per_net * self.num_nets,
            self.max_num_pins_per_net * self.num_nets + 1,
        )

        self.total_num_pins = min(
            total_num_pins, self.total_area_covered_by_all_components
        )

    def allocate_pins_to_nets(self):  # noqa: max-complexity: 13
        r"""Allocates pins to nets.

        Allocates pins to nets by first sampling unnormalized probabilities
        for the probabilitiy of a pin being allocated to a net. This sampling is
        done by sampling from a normal distribution with mean :math:`\frac{1}{\text{num_nets}}`
        and standard deviation :math:`\frac{1}{\text{net_distribution} + 1}`, where the
        addition of 1 to net_distribution is to ensure the sampled values are
        positive. The sampled values are then normalized using softmax to get
        the probabilities of allocating a pin to each net.

        Then the number of pins allocated to each net is sampled using a
        multinomial distribution with the probabilities calculated above and
        the total number of pins to be allocated to nets. This determines the
        allocation of pins to nets.
        """
        # Sample from a normal distribution with mean 1/num_nets and
        # standard deviation 1/(net_distribution+1) to get the percentage of pins
        # allocated to each net
        samples_for_pct_pins_per_net = np.random.normal(
            1 / self.num_nets, 1 / (self.net_distribution + 1), self.num_nets
        )
        # Apply softmax to samples_for_pct_pins_per_net to get the probabilities
        # of allocating pins to each net.
        probs_for_nets = np.exp(samples_for_pct_pins_per_net) / np.sum(
            np.exp(samples_for_pct_pins_per_net)
        )

        # An array of size num_nets, where each element is the number of
        # pins allocated to the corresponding net.
        # Assign net ids to pins and create dictionary net_pins to store pins for each net
        pin_id = 0
        self.net_pins = {}

        for i in range(self.num_nets):
            for j in range(self.min_num_pins_per_net):
                self.pins[j + pin_id].net_id = i
            pin_id += self.min_num_pins_per_net
            self.net_pins[i] = self.pins[pin_id - self.min_num_pins_per_net : pin_id]

        if (self.max_num_pins_per_net > self.min_num_pins_per_net) and (
            self.total_num_pins - self.min_num_pins_per_net * self.num_nets
        ) > 0:
            net_allocation = sample_truncated_multinomial(
                self.num_nets,
                self.total_num_pins - self.min_num_pins_per_net * self.num_nets,
                probs_for_nets,
                min(
                    self.max_num_pins_per_net - self.min_num_pins_per_net,
                    self.total_num_pins - self.min_num_pins_per_net * self.num_nets,
                ),
            )

            for i in range(self.num_nets):
                for j in range(net_allocation[i]):
                    self.pins[j + pin_id].net_id = i
                if net_allocation[i] > 0:
                    pin_id += net_allocation[i]
                    self.net_pins[i].extend(
                        self.pins[pin_id - net_allocation[i] : pin_id]
                    )

    def allocate_pins_to_components(self):
        r"""Allocates pins to components.

        The number of components that each net will cover is calculated by:
            - First finding the maximum between :math:`1` and
                :math:`\frac{(\text{pin_spread} + 1) * \text{num_components}}{10}`.
            - Then finding the minimum between the value calculated above and
                :math:`\text{num_components}`.

        The pin spread controls how evenly the pins are distributed across the
        components. A higher pin spread means that the pins are more evenly
        distributed across the components. Note that the min and max operations
        are used to ensure that the number of components that each net will
        cover is at least 1 and at most the total number of components.

        This value is then used to allocate pins to components by looping through
        all nets and using the allocate_pins_to_components_for_net method.
        """
        # Number of components that each net will cover based on complexity (pin_spread)
        self.num_components_w_pins = min(
            int(((self.pin_spread) / 10) * self.num_components) + 1, self.num_components
        )
        # Create dictionary of components and number of available spaces for pins
        components_available_space = {}
        for component in self.components:
            components_available_space[component.comp_id] = self.components[
                component.comp_id
            ].area
        # Allocate pins to components
        for net in range(self.num_nets):
            components_available_space = self.allocate_pins_to_components_for_net(
                net, components_available_space, self.num_components_w_pins
            )
        # Update self.pins using self.net_pins
        self.pins = []
        for net in range(self.num_nets):
            self.pins.extend(self.net_pins[net])

    def allocate_pins_to_components_for_net(  # noqa: max-complexity: 15
        self,
        net: int,
        components_available_space: Dict[int, int],
        num_components_w_pins: int,
    ) -> Dict[int, int]:
        """Allocates pins to components for a given net.

        The algorithm to allocate pins to components for a given net is as
        follows:

            - Order the components_available_space dictionary by number of
                available spaces for pins.
            - Update num_components_w_pins to ensure that all pins are assigned.
            - Loop through all components and assign pins to components until
                all pins are assigned.

        Args:
            net (int): The net to allocate pins to.
            components_available_space (Dict[int, int]): A dictionary of
                components and the number of available spaces for pins.
            num_components_w_pins (int): The number of components to
                allocate pins to.

        Returns:
            Dict[int, int]: A dictionary of components and the number of
                available spaces for pins.
        """
        num_pins_unassigned = len(self.net_pins[net])

        # Order components_available_space dictionary by number of available
        # spaces for pins
        components_available_space = dict(
            sorted(
                components_available_space.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # Update num_components_w_pins to ensure that all pins are assigned
        total_components_space = 0
        num_components_w_pins -= 1
        while total_components_space < num_pins_unassigned:
            total_components_space = 0
            num_components_w_pins += 1
            components_to_assign_pins = list(components_available_space.keys())[
                :num_components_w_pins
            ]
            for component in components_to_assign_pins:
                component_space = components_available_space[component]
                total_components_space += component_space

        pin_id = 0
        while num_pins_unassigned > 0:
            # Take the first num_components_w_pins components and allocate pins to them
            components_to_assign_pins = list(components_available_space.keys())[
                :num_components_w_pins
            ]

            total_space_available = sum(
                components_available_space[component]
                for component in components_to_assign_pins
            )

            # Allocate pins in net to components proportionally to their available space
            num_pins_per_component = np.random.multinomial(
                num_pins_unassigned,
                np.array(
                    [
                        components_available_space[component] / total_space_available
                        for component in components_to_assign_pins
                    ]
                ),
            )
            for component, num_pins in zip(
                components_to_assign_pins, num_pins_per_component
            ):
                # If there are more pins to be assigned than available space,
                # assign all available space
                if components_available_space[component] < num_pins:
                    num_pins = components_available_space[component]
                components_available_space[component] -= num_pins

                # Assign component ids to pins
                for pin in range(num_pins):
                    self.net_pins[net][pin + pin_id].component_id = component
                pin_id += num_pins
                # update num_pins_unassigned
                num_pins_unassigned -= num_pins

        return components_available_space

    def get_net_pin_positions(self) -> Dict[int, List[Tuple[int, int]]]:
        """Returns a dictionary of net ids and the positions of their pins.

        Returns:
            Dict[int, List[Tuple[int, int]]]: A dictionary of net ids and their corresponding pin positions.
        """
        net_pin_positions = {}
        # Loop through pins for each net
        for net in self.net_pins.keys():
            # Create key value pair for net and pin positions.
            net_pin_positions[net] = [
                (pin.absolute_x, pin.absolute_y) for pin in self.net_pins[net]
            ]
        return net_pin_positions

    @staticmethod
    def get_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns the centroid of a set of points.

        Args:
            points (List[Tuple[int, int]]): A list of points.

        Returns:
            Tuple[int, int]: The centroid of the points.
        """
        points_arr = np.array(points)
        cx, cy = np.mean(points_arr, axis=0)
        return (cx, cy)

    def route_pins_centroid(
        self,
    ) -> List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Routes pins to the centroid of the pins for each net.

        Returns:
            List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]: A list of the routes
                for each net, where each route is a list of point pairs. The pairs
                are the start and end points of each line segment (connection)
                in the route.
        """
        net_pin_positions = self.get_net_pin_positions()
        route_point_pairs = []

        # Loop through sets of pin positions for each net
        for positions in net_pin_positions.values():
            if len(positions) == 2:
                # If there are only two pins, route directly between them
                net_route = [(positions[0], positions[1])]

            else:
                # Get centroid
                centroid = self.get_centroid(positions)

                # Add each pair of line endpoints to net route
                net_route = [(position, centroid) for position in positions]
            route_point_pairs.append(net_route)

        return route_point_pairs

    def pin_outlier(self, pin_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns the pin that is furthest away from the centroid of a set of pins.

        Args:
            pin_positions (List[Tuple[int, int]]): A list of pin positions.

        Returns:
            Tuple[int, int]: The pin that is furthest away from the centroid.
        """
        centroid = self.get_centroid(pin_positions)
        distances = [
            np.linalg.norm(np.array(pin) - np.array(centroid)) for pin in pin_positions
        ]
        return pin_positions[np.argmax(distances)]

    @staticmethod
    def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate the euclidean distance between two points.

        Args:
            point1 (Tuple[int, int]): The first point.
            point2 (Tuple[int, int]): The second point.

        Returns:
            float: The euclidean distance between the two points.
        """
        point_1 = np.array(point1)
        point_2 = np.array(point2)
        return np.linalg.norm(point_1 - point_2)

    def beam_search(
        self,
        start_point: Tuple[int, int],
        points: List[Tuple[int, int]],
        beam_width: int = 2,
    ) -> List[Tuple[int, int]]:
        """Find shortest path that visits all points using beam search.

        The algorithm works as follows:

            - For the given start point, find the beam_width number of points
                with the smallest distance to the start point.
            - For each of these beam_width points, find the beam_width number of
                points with the smallest distance to that point.
            - Repeat this process until all points have been visited and
                return the shortest path from all the beam_width number of paths.

        This algorithm is implemented using a priority queue which stores
        the total distance of the path, the path itself, and the set of points
        that have been visited. The priority queue is sorted by the total distance
        of the path.

        Args:
            start_point (Tuple[int, int]): The starting point for the algorithm.
            points (List[Tuple[int, int]]): The points to visit (not including start_point).
                beam_width (int): The beam width. Defaults to 2.

        Returns:
            List[Tuple[int, int]]: The shortest path that visits all points.
        """
        # Initialize the priority queue with the starting point and an empty set of visited points
        points_to_visit = set(points)
        queue: List[Tuple[float, List[Tuple[int, int]], Set[Tuple[int, int]]]] = [
            (0, [start_point], set())
        ]

        while queue:
            new_queue: List[
                Tuple[float, List[Tuple[int, int]], Set[Tuple[int, int]]]
            ] = []
            # Loop through the beam width number of paths with the smallest total distance
            for _ in range(min(beam_width, len(queue))):
                # Get the path with the smallest total distance so far
                priority, path, visited = heapq.heappop(queue)
                current = path[-1]

                # If all points have been visited, return the path
                if visited == set(points_to_visit):
                    return path

                # Sort the neighbors by distance to the current point
                neighbors = sorted(
                    points_to_visit - visited,
                    key=lambda x: self.euclidean_distance(current, x),
                )
                priorities = [priority for _ in range(len(neighbors[:beam_width]))]

                # Push the beam width nearest neighbour paths to the queue.
                for ind, neighbor in enumerate(neighbors[:beam_width]):
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}

                    # Add the new path to the queue, using the total distance as priority
                    priorities[ind] += self.euclidean_distance(neighbor, current)
                    heapq.heappush(new_queue, (priorities[ind], new_path, new_visited))
            queue = new_queue
        return path

    def route_pins_beam_search(
        self, beam_width: int = 2
    ) -> List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Routes pins using beam search.

        Route the pins for each net using the beam_search method,
        where points represent the positions of the pins. The starting
        point is the pin that is furthest away from the centroid of the
        pins in the net.

        Args:
            beam_width (int): The width of the beam. Defaults to 2.

        Returns:
            List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]: A list of the routes
                for each net, where each route is a list of point pairs. The pairs
                are the start and end points of each line segment (connection)
                in the route.
        """
        # Get start pin position
        pin_positions = self.get_net_pin_positions()
        route_point_pairs = []
        for net in pin_positions:
            # Get start pin
            start = self.pin_outlier(pin_positions[net])
            pin_positions[net].remove(start)

            # Get shortest path and line segments
            shortest_path = self.beam_search(start, pin_positions[net], beam_width)
            line_segments = [
                (shortest_path[i], shortest_path[i + 1])
                for i in range(len(shortest_path) - 1)
            ]
            route_point_pairs.append(line_segments)

        return route_point_pairs

    def get_all_relative_coordinates_in_component(self, component: Component):
        """Gets all relative coordinates (relative to the component's top-left corner)
        in a component.

        Args:
            component (Component): A Component object.

        Returns:
            List[Tuple[int, int]]: List of relative coordinates.
        """
        coordinates = []
        for x in range(component.h):
            for y in range(component.w):
                coordinates.append((x, y))
        return coordinates

    def place_pins_on_component(self, component: Component):
        """Place pins on a component.

        Randomly choose a relative coordinate in the component and assign it to
        the pin.

        Args:
            component (Component): A Component object.
        """
        component_relative_coordinates = self.get_all_relative_coordinates_in_component(
            component
        )

        for pin in component.pins:
            # randomly choose a relative coordinate in the component
            relative_coordinate = random.choice(component_relative_coordinates)
            # remove relative_coordinate from component_relative_coordinates
            component_relative_coordinates.remove(relative_coordinate)
            pin.relative_x = relative_coordinate[0]
            pin.relative_y = relative_coordinate[1]

    def update_placement_mask(self, component: Component):
        """Updates the placement_mask array for a component that has been placed.

        Args:
            component (Component): A Component object.
        """
        self.placement_mask[component.comp_id] = 2.0

    def update_all_components_feature(self, components: List[Component]):
        """Updates the all_components_feature given a list of components.

        Args:
            components (List[Component]): List of components.

        Notes:
            This method is called when the environment is reset.
        """
        for component in components:
            feature = component.calculate_feature(
                self.area, self.max_num_pins_per_component
            )

            self.all_components_feature[component.comp_id, :] = feature

    def update_all_pins_feature(self, components: List[Component]):
        """Updates the all_pins_feature given a list of components with
        pins placed on them.

        Args:
            components (List[Component]): List of components.

        Notes:
            This method is called when the environment is reset.
        """
        for component in components:
            for pin in component.pins:
                pin_num_feature, pin_cat_feature = pin.calculate_feature(
                    component.position[0], component.position[1]
                )
                self.all_pins_num_feature[pin.pin_id, :] = pin_num_feature
                self.all_pins_cat_feature[pin.pin_id, :] = pin_cat_feature

    def reset(self, verbose: bool = False, *args: Any, **kwargs: Any) -> dict:
        """Resets the environment.

        At each reset, a list of components with pins placed on them,
        and a list of nets with allocated pins are generated. The environment
        state is initialized and following reset components are stepped through
        in the order that they were generated.

        Args:
            verbose (bool): Whether to print the grid and action mask after the
                environment is reset.

        Returns:
            observation (dict): The initial observation.
        """
        # Generate instances of components
        self.grid = np.zeros((self.height, self.width))
        self.pin_grid = np.zeros((self.height, self.width, self.max_num_nets + 1))
        self.net_pins = {}
        self.generate_instances()
        self.draw_components()

        # Initialize masks and all_components_features
        self.all_components_feature = np.zeros(
            (self.max_num_components, 5 + self.max_num_pins_per_component),
            dtype=np.float64,
        )
        self.all_pins_num_feature = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component + 1, 4)
        )
        self.all_pins_cat_feature = np.zeros(
            (self.max_num_components * self.max_num_pins_per_component + 1, 2)
        )
        self.all_pins_cat_feature[-1, :] = -1
        self.current_component = self.components[0]
        self.current_component_id = self.current_component.comp_id
        self.action_mask = self.compute_action_mask(self.current_component)
        # Update masks and features
        self.placement_mask = np.ones(self.max_num_components)
        self.placement_mask[len(self.components) :] = 0.0
        self.placement_mask[self.current_component_id] = 3.0
        self.update_all_components_feature(self.components)
        self.update_all_pins_feature(self.components)

        # Show the grid and action mask
        if verbose:
            print("Grid:")
            print(self.grid, end="\n\n")
            print("Action mask:")
            print(self.action_mask, end="\n\n")

        state = {
            "grid": self.grid.copy(),
            "pin_grid": self.pin_grid.copy(),
            "component_grid": self.component_grid.copy(),
            "action_mask": self.action_mask.copy(),
            "all_components_feature": self.all_components_feature.copy(),
            "placement_mask": self.placement_mask.copy(),
            "all_pins_num_feature": self.all_pins_num_feature.copy(),
            "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
        }

        return state

    def step(  # noqa: max-complexity: 15
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
                environment is reset.

        Returns:
            observation (dict): The observation after the action is taken.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """

        orientation, x, y = action
        self.actions.append(action)
        valid_action = self.validate_action(orientation, x, y)

        if valid_action:
            # Place the component on the grid
            self.update_grid(self.current_component, orientation, x, y)
            self.current_component.place_component(orientation, x, y)

            feature = self.current_component.calculate_feature(
                self.area, self.max_num_pins_per_component
            )
            # Update the all components feature
            self.all_components_feature[self.current_component.comp_id, :] = feature
            # Update all pins feature
            self.update_all_pins_feature(self.components)

            self.draw_pins()
            # Update the placement mask
            self.update_placement_mask(self.current_component)
            # Set current component to next component
            if self.current_component.comp_id + 1 < len(self.components):
                self.current_component = self.components[
                    self.current_component.comp_id + 1
                ]
                self.current_component_id = self.current_component.comp_id
                self.placement_mask[self.current_component.comp_id] = 3.0
            else:
                self.current_component = Component(-1, -1, -1, [])
                self.current_component_id = -1

            # Compute action mask for next component
            if self.current_component.comp_id != -1:
                self.action_mask = self.compute_action_mask(self.current_component)
            else:
                self.action_mask = np.zeros((4, self.height, self.width))

            # Calculate the reward
            done = self.compute_if_done()
            if not done:
                reward = 0.0
                info = {}
            else:
                reward = self.find_reward()
                info = {
                    "wirelength": self.reward_wirelength,
                    "num_intersections": self.reward_intersection,
                }

            state = {
                "grid": self.grid.copy(),
                "pin_grid": self.pin_grid.copy(),
                "component_grid": self.component_grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "placement_mask": self.placement_mask.copy(),
                "all_pins_num_feature": self.all_pins_num_feature.copy(),
                "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
            }

            # Print the grid and action mask if verbose
            if verbose:
                print("Grid:")
                print(self.grid, end="\n\n")
                print("Action mask:")
                print(self.action_mask, end="\n")
                print("Reward:", reward, end="\n\n")

            return state, reward, done, info

        else:
            self.draw_pins()
            # print('invalid')
            state = {
                "grid": self.grid.copy(),
                "pin_grid": self.pin_grid.copy(),
                "component_grid": self.component_grid.copy(),
                "action_mask": self.action_mask.copy(),
                "all_components_feature": self.all_components_feature.copy(),
                "placement_mask": self.placement_mask.copy(),
                "all_pins_num_feature": self.all_pins_num_feature.copy(),
                "all_pins_cat_feature": self.all_pins_cat_feature.copy(),
            }
            reward = self.find_reward()
            info = {
                "wirelength": self.reward_wirelength,
                "num_intersections": self.reward_intersection,
            }
            return state, reward, True, info

    def draw_pins(self):
        """Create pin grid from the current grid and pin placements."""
        curr_grid = self.grid.copy()
        for net in self.net_pins.keys():
            for pin in self.net_pins[net]:
                if int(pin.absolute_x) == -1 or int(pin.absolute_y) == -1:
                    continue
                curr_grid[int(pin.absolute_x), int(pin.absolute_y)] = net + 2

        pin_grid = curr_grid.astype(int)
        pin_grid_one_hot = np.eye(self.max_num_nets + 2)[pin_grid]

        self.pin_grid = pin_grid_one_hot[:, :, 1:]

    def draw_components(self):
        """Create component grid from the current grid and component placements."""
        self.component_grid = np.zeros(
            (
                len(self.components),
                self.max_component_h,
                self.max_component_w,
                self.max_num_nets + 1,
            )
        )
        for component in self.components:
            for pin in component.pins:
                if pin.relative_x == -1 or pin.relative_y == -1:
                    continue

                self.component_grid[
                    component.comp_id, pin.relative_x, pin.relative_y, pin.net_id + 1
                ] = 1
        self.component_grid[:, :, :, 0] = np.ones(
            (len(self.components), self.max_component_h, self.max_component_w)
        )

    def validate_action(self, orientation: int, x: int, y: int) -> bool:
        """Validates the given action.

        The action is invalid if it would cause the component to go out of
        bounds or if any of cells that the component would occupy are already
        occupied. Note that the action mask both checks if the action is out
        of bounds and if any of the cells are already occupied.

        Args:
            x (int): The x coordinate of the top left corner of the component.
            y (int): The y coordinate of the top left corner of the component.
            orientation (int): The orientation of the component (0, 1, 2, or 3),
                where 0 means the component is placed in its original orientation,
                1 means the component is rotated 90 degrees clockwise from its
                original orientation, 2 means the component is rotated 180 degrees
                clockwise from its original orientation, and 3 means the component
                is rotated 270 degrees clockwise from its original orientation.

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
            component (Component): The component to place on the grid.
            x (int): The x coordinate of the top left corner of the component.
            y (int): The y coordinate of the top left corner of the component.
            orientation (int): The orientation of the component (0, 1, 2, or 3),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is rotated 90 degrees (b x a),
                2 means the component is rotated 180 degrees (a x b), and 3 means the
                component is rotated 270 degrees (b x a).

        Raises:
            Exception: If the orientation is not 0, 1, 2, or 3.
        """
        # Set the placement height and width based on the orientation
        if orientation in [0, 2]:
            placement_height = component.h
            placement_width = component.w
        elif orientation in [1, 3]:
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
            orientation (int): The orientation of the component (0, 1, 2, or 3),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is rotated 90 degrees (b x a),
                2 means the component is rotated 180 degrees (a x b), and 3 means the
                component is rotated 270 degrees (b x a).

        Returns:
            rows_to_mask (list[int]): The rows to mask.
            cols_to_mask (list[int]): The columns to mask.

        Raises:
            Exception: if the orientation is not 0, 1, 2, or 3.
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
            component (Component): The component to place on the grid.
            orientation (int): The orientation of the component (0, 1, 2, or 3),
                where 0 means the component is placed in its original (a x b)
                orientation and 1 means the component is rotated 90 degrees (b x a),
                2 means the component is rotated 180 degrees (a x b), and 3 means the
                component is rotated 270 degrees (b x a).

        Returns:
            action_mask_orientation (np.ndarray): The component's action mask
                for the given orientation.

        Raises:
            Exception: If the orientation is not 0, 1, 2, or 3.
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
        action_mask = np.zeros((4, self.height, self.width))

        # Compute the action mask for each orientation
        action_mask[0] = self.compute_action_mask_orientation(component, 0)
        action_mask[1] = self.compute_action_mask_orientation(component, 1)
        action_mask[2] = np.copy(action_mask[0])
        action_mask[3] = np.copy(action_mask[1])
        return action_mask

    def compute_if_done(self) -> bool:
        """Computes whether the episode is terminated.

        The episode is terminated when it is not possible to place the
        next component on the grid or all components have been placed.

        Returns:
            bool: Whether the episode is terminated.
        """
        # Check if there are no more components to place
        if self.current_component.comp_id != -1:
            # Check if there are no more actions to take
            return bool(np.all(self.action_mask == 0))
        return True

    def __str__(self) -> str:
        """Returns a string representation of the grid."""
        return str(self.grid)
