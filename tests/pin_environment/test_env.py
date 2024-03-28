import numpy as np
import math
from environment.dummy_env_rectangular_pin import Component, Pin
from collections import defaultdict


def test_update_grid_rotated(env_fixed_components_reset):
    """Test the update grid method for an environment with grid
    size 6x6. Test for placing components in the rotated orientation."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 2, 1, 0, 0)]
    env.components = [
        Component(1, 2, 0, pins[0:1]),
        Component(3, 2, 1, pins[1:]),
    ]

    # Place component in rotated orientation
    env.update_grid(env.components[0], 1, 0, 0)
    assert env.grid[0, 0] == 1
    assert env.grid[1, 0] == 1


def test_update_grid_original(env_fixed_components_reset):
    """Test the update grid method for an environment with grid
    size 6x6. Test for placing components in the non-rotated orientation."""
    env = env_fixed_components_reset

    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 2, 1, 0, 0)]
    env.components = [Component(1, 2, 0, pins[0:1]), Component(3, 2, 1, pins[1:])]
    env.current_component = env.components[0]

    # Place component in rotated orientation
    env.update_grid(env.components[0], 0, 0, 0)
    assert env.grid[0, 0] == 1
    assert env.grid[0, 1] == 1


def test_intersection_0(env_fixed_components_reset):
    """Test the is_intersect method for when 2 lines do
    not intersect."""
    env = env_fixed_components_reset
    assert not env.is_intersect(((1, 1), (3, 3)), ((1, 3), (1, 5)))


def test_intersection_1(env_fixed_components_reset):
    """Test the is_intersect method for when 2 lines do
    intersect."""
    env = env_fixed_components_reset
    assert env.is_intersect(((1, 1), (3, 3)), ((1, 3), (2, 1)))


def test_find_num_intersection(env_fixed_components_reset):
    """Test the find_num_intersection to check that
    the number of intersections of a route is found correctly."""
    env = env_fixed_components_reset
    assert (
        env.find_num_intersection(
            [
                [((1, 1), (3, 3))],
                [((2, 1), (0, 3))],
                [((2, 3), (0, 1))],
                [((3, 2), (1, 3))],
            ]
        )
        == 4
    )


def test_lowest_num_intersections(env_fixed_components_reset):
    """Test the lowest_num_intersections method to check if it finds the
    lowest number of intersections given two routes."""
    env = env_fixed_components_reset
    assert env.lowest_num_intersections(
        [
            [
                [((1, 1), (3, 3))],
                [((2, 1), (0, 3))],
                [((2, 3), (0, 1))],
                [((3, 2), (1, 3))],
            ],
            [[((4, 4), (3, 5))], [((3, 4), (4, 5))]],
        ]
    ) == (1, 1)


def test_upper_bound_intersections(env_upper_bound_intersections):
    """ " Test that the upper bound for the number of intersections
    for a gird is correctly found."""
    env = env_upper_bound_intersections
    expected_upper_bound = 48
    assert env.max_num_intersections == expected_upper_bound


def test_get_centroid(env_10_10):
    """Test that the centroid of a set of pins is
    calculated correctly."""
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = env_10_10
    assert env.get_centroid(points) == (0.5, 0.5)


def test_route_pins_centroid(env_10_10, component_3x3_2_pins, component_3x3_3_pins):
    """Test that the method for routing pins to the centroid
    of the other pins works correctly."""
    env = env_10_10
    component1 = component_3x3_2_pins
    component2 = component_3x3_3_pins

    # Create dictionary for the pins of each net
    net_pins = {}
    net_pins["net1"] = component1.pins
    net_pins["net2"] = component2.pins
    env.net_pins = net_pins

    # Route the pins to the centroid
    route = env.route_pins_centroid()
    assert route == [
        [((0, 0), (0, 1))],
        [((2, 2), (3.0, 3.0)), ((3, 3), (3.0, 3.0)), ((4, 4), (3.0, 3.0))],
    ]


def test_pin_outlier(env_fixed_components_reset):
    """Test that the outlier pin for the centroid method is
    found correctly."""
    env = env_fixed_components_reset
    points = [(0, 0), (0, 1), (1, 0), (3, 3)]
    calculated_outlier = env.pin_outlier(points)
    assert calculated_outlier == (3, 3)


def test_beam_search_width_full(env_fixed_components_reset):
    """Test the beam_search method for finding the shortest path
    for a set of points for when the beam width is equal to the
    number of points. This should return the shortest path."""
    env = env_fixed_components_reset
    start, points = (0, 0), [(2, 2), (0, 1), (1, 0), (1, 1)]
    path = env.beam_search(start, points, 4)
    assert path == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]


def test_beam_search_width_2(env_fixed_components_reset):
    """Test the beam_search method for finding the shortest path
    for a set of points for when the beam width is 2."""
    env = env_fixed_components_reset
    start, points = (0, 0), {(2, 2), (0, 1), (1, 0), (1, 1)}
    path = env.beam_search(start, points, 2)

    assert path == [(0, 0), (0, 1), (1, 1), (1, 0), (2, 2)]


def test_beam_search_route_pins(
    env_10_10, component_3x3_2_pins_beam, component_3x3_5_pins_beam
):
    """Test that the beam search method of routing pins works correctly."""
    env = env_10_10
    component1 = component_3x3_2_pins_beam
    component2 = component_3x3_5_pins_beam

    # Create dictionary for the pins of each net
    net_pins = {}
    net_pins["net1"] = component1.pins
    net_pins["net2"] = component2.pins
    env.net_pins = net_pins

    # Route the pins by beam search
    route = env.route_pins_beam_search(beam_width=2)
    assert route == [
        [((3, 3), (3, 4))],
        [((2, 2), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0))],
    ]


def test_find_wirelength(env_fixed_components_reset):
    """Test the find_wirelength method finds the correct
    wirelength for a given route."""
    route = [[((3, 1), (2, 2))], [((1, 2), (2, 2))], [((3, 3), (2, 2))]]
    assert np.isclose(
        env_fixed_components_reset.find_wirelength(route), 1 + 2 * np.sqrt(2)
    )


def test_upper_bound_wirelength(env_fixed_components_reset):
    """Test the upper_bound_wirelength method correctly
    finds the upper bound on the maximum wirelength for
    a given grid size."""
    env = env_fixed_components_reset
    expected_upper_bound = 0.5 * 8 * math.sqrt(72)
    assert np.isclose(env.max_wirelength, expected_upper_bound)


def test_euclidean_distance(env_fixed_components_reset):
    """Test the euclidean distance function"""
    assert env_fixed_components_reset.euclidean_distance((0, 0), (1, 1)) == math.sqrt(2)


def test_find_reward_beam(
    env_10_10_reward,
    component_3x3_2_pins_reward_1,
    component_3x3_2_pins_reward_2,
    component_2x1_1_pin_reward_1,
    component_2x1_1_pin_reward_2,
    component_2x2_1_pin_reward,
):
    """Test that the reward for when we are using beam search
    for routing is calculated correctly."""
    env = env_10_10_reward
    env.reward_type = "beam"

    # Set components
    component1 = component_3x3_2_pins_reward_1
    component2 = component_3x3_2_pins_reward_2
    component3 = component_2x1_1_pin_reward_1
    component4 = component_2x1_1_pin_reward_2
    component5 = component_2x2_1_pin_reward
    components = [component1, component2, component3, component4, component5]

    # Create dictionary for the pins of each net
    env.net_pins = defaultdict(list)
    for component in components:
        for pin in component.pins:
            env.net_pins[pin.net_id].append(pin)

    # Route the pins by beam search and find the reward
    env.current_component = Component(-1, -1, -1, [])
    reward = env.find_reward()

    # Get expected wirelength
    expected_wirelength = (
        np.sqrt(26) + np.sqrt(13) + np.sqrt(17) + np.sqrt(10) + np.sqrt(5)
    )
    wirelength_normalizing_factor = 20

    # Get expected number of intersections
    expected_num_intersections = 1
    avg_total_num_pins_comp_size = (
        np.mean([env.min_component_h, env.max_component_h])
        * np.mean([env.min_component_w, env.max_component_w])
        * np.mean([env.min_num_components, env.max_num_components])
    )
    avg_total_num_pins_net = np.mean(
        [env.min_num_pins_per_net, env.max_num_pins_per_net]
    ) * np.mean([env.min_num_nets, env.max_num_nets])
    intersections_normalizing_factor = np.min(
        [avg_total_num_pins_comp_size, avg_total_num_pins_net]
    )

    # Expected reward
    expected_reward = -0.5 * (
        expected_wirelength / wirelength_normalizing_factor
        + expected_num_intersections / intersections_normalizing_factor
    )
    assert np.isclose(reward, expected_reward)


def test_find_reward_centroid(
    env_10_10_reward,
    component_3x3_2_pins_reward_1,
    component_3x3_2_pins_reward_2,
    component_2x1_1_pin_reward_1,
    component_2x1_1_pin_reward_2,
    component_2x2_1_pin_reward,
):
    """Test that the reward for when we are using the centroid method
    for routing is calculated correctly."""
    env = env_10_10_reward
    env.reward_type = "centroid"

    # Set components
    component1 = component_3x3_2_pins_reward_1
    component2 = component_3x3_2_pins_reward_2
    component3 = component_2x1_1_pin_reward_1
    component4 = component_2x1_1_pin_reward_2
    component5 = component_2x2_1_pin_reward
    components = [component1, component2, component3, component4, component5]

    # Create dictionary for the pins of each net
    env.net_pins = defaultdict(list)
    for component in components:
        for pin in component.pins:
            env.net_pins[pin.net_id].append(pin)

    # Route the pins by beam search and find the reward
    env.current_component = Component(-1, -1, -1, [])
    reward = env.find_reward()

    # Get expected wirelength
    expected_wirelength = (
        13 / 3
        + np.sqrt(13) / 3
        + np.sqrt(130) / 3
        + np.sqrt(41) / 2
        + 3 / 2
        + np.sqrt(61) / 2
        + np.sqrt(13) / 2
    )
    wirelength_normalizing_factor = 20

    # Get expected number of intersections
    expected_num_intersections = 2
    avg_total_num_pins_comp_size = (
        np.mean([env.min_component_h, env.max_component_h])
        * np.mean([env.min_component_w, env.max_component_w])
        * np.mean([env.min_num_components, env.max_num_components])
    )
    avg_total_num_pins_net = np.mean(
        [env.min_num_pins_per_net, env.max_num_pins_per_net]
    ) * np.mean([env.min_num_nets, env.max_num_nets])
    intersections_normalizing_factor = np.min(
        [avg_total_num_pins_comp_size, avg_total_num_pins_net]
    )

    # Expected reward
    expected_reward = -0.5 * (
        expected_wirelength / wirelength_normalizing_factor
        + expected_num_intersections / intersections_normalizing_factor
    )
    assert np.isclose(reward, expected_reward)


def test_find_reward_both(
    env_10_10_reward,
    component_3x3_2_pins_reward_1,
    component_3x3_2_pins_reward_2,
    component_2x1_1_pin_reward_1,
    component_2x1_1_pin_reward_2,
    component_2x2_1_pin_reward,
):
    """Test that the reward for when we are using both beam search
    and the centroid method for routing is calculated correctly."""
    env = env_10_10_reward
    env.reward_type = "both"

    # Set components
    component1 = component_3x3_2_pins_reward_1
    component2 = component_3x3_2_pins_reward_2
    component3 = component_2x1_1_pin_reward_1
    component4 = component_2x1_1_pin_reward_2
    component5 = component_2x2_1_pin_reward
    components = [component1, component2, component3, component4, component5]

    # Create dictionary for the pins of each net
    env.net_pins = defaultdict(list)
    for component in components:
        for pin in component.pins:
            env.net_pins[pin.net_id].append(pin)

    # Route the pins by beam search and find the reward
    env.current_component = Component(-1, -1, -1, [])
    reward = env.find_reward()

    # Get expected wirelength
    expected_wirelength = (
        np.sqrt(26) + np.sqrt(13) + np.sqrt(17) + np.sqrt(10) + np.sqrt(5)
    )
    wirelength_normalizing_factor = 20

    # Get expected number of intersections
    expected_num_intersections = 1
    avg_total_num_pins_comp_size = (
        np.mean([env.min_component_h, env.max_component_h])
        * np.mean([env.min_component_w, env.max_component_w])
        * np.mean([env.min_num_components, env.max_num_components])
    )
    avg_total_num_pins_net = np.mean(
        [env.min_num_pins_per_net, env.max_num_pins_per_net]
    ) * np.mean([env.min_num_nets, env.max_num_nets])
    intersections_normalizing_factor = np.min(
        [avg_total_num_pins_comp_size, avg_total_num_pins_net]
    )

    # Expected reward
    expected_reward = -0.5 * (
        expected_wirelength / wirelength_normalizing_factor
        + expected_num_intersections / intersections_normalizing_factor
    )
    assert np.isclose(reward, expected_reward)


def test_find_reward_not_all_placed(env_10_10_reward):
    """Test that the reward is calculated correctly when not all components
    have been able to be placed on the grid."""
    env = env_10_10_reward
    env.reset()

    # Find reward
    reward = env.find_reward()
    expected_reward = -0.5 * 2 * math.sqrt(2) - 0.5 * 24 / 8
    assert np.isclose(reward, expected_reward)


def test_sample_num_components(env_low_complexity):
    """Test the sampling of the number of components on a grid of size 6x6"""
    env = env_low_complexity
    for _ in range(10):
        env.sample_num_components()
        assert env.num_components <= env.max_num_components
        assert env.num_components >= env.min_num_components


def test_sample_num_nets(env_low_complexity):
    """Test that the number of nets is sampled correctly."""
    env = env_low_complexity

    env.generate_components()
    env.sample_num_nets()
    assert env.num_nets == min(
        env.min_num_nets, int(env.total_area_covered_by_all_components / 2)
    )

    env.net_distribution = 5
    env.pin_spread = 5
    env.generate_components()
    env.sample_num_nets()
    assert (
        min(env.min_num_nets, int(env.total_area_covered_by_all_components / 2))
        <= env.num_nets
    )
    assert env.num_nets <= env.max_num_nets

    env.net_distribution = 9
    env.pin_spread = 9
    env.generate_components()
    env.sample_num_nets()
    assert env.num_nets == min(
        env.max_num_nets, int(env.total_area_covered_by_all_components / 2)
    )


def test_sample_total_num_pins(env_low_complexity):
    """Test the sampling of the total number of pins on a grid of size 6x6"""
    env = env_low_complexity
    for _ in range(10):
        env.generate_components()
        total_area_covered_by_all_components = 0
        for component in env.components:
            total_area_covered_by_all_components += component.area
        env.sample_num_nets()
        env.sample_total_num_pins()
        assert env.total_num_pins == total_area_covered_by_all_components or (
            env.total_num_pins <= env.max_num_pins_per_net * env.num_nets
            and env.total_num_pins >= env.min_num_pins_per_net * env.num_nets
        )


def test_allocate_pins_to_nets(env_low_complexity):
    """Test the allocation of pins to nets on a grid of size 6x6."""
    env = env_low_complexity

    for _ in range(10):
        env.generate_components()
        env.sample_num_nets()
        env.sample_total_num_pins()
        env.generate_pins()
        env.allocate_pins_to_nets()
        total_pins = 0
        for _, value in env.net_pins.items():
            total_pins += len(value)
            assert len(value) > 0
        assert total_pins == env.total_num_pins


def test_allocate_pins_to_components(env_10_10_low_complexity):
    """Test that pins are correctly allocated to components."""
    env = env_10_10_low_complexity

    for i in range(10):
        env.net_distribution = i
        env.pin_spread = i
        env.generate_components()
        env.sample_num_nets()
        env.sample_total_num_pins()
        env.generate_pins()
        env.allocate_pins_to_nets()

        env.allocate_pins_to_components()

        assert env.num_components_w_pins >= 1
        assert env.num_components_w_pins <= env.num_components

        # Check that all pins are allocated to a component (none of the pins in self.pins have component_id = -1)
        assert len(env.pins) == env.total_num_pins
        assert len(env.pins) == len([pin for pin in env.pins if pin.component_id != -1])


def test_allocate_pins_to_components_for_net(env_10_10_low_complexity):
    """Test allocation of pins to components for a single net."""
    env = env_10_10_low_complexity
    env.num_nets = 3
    env.num_components = 5
    env.total_num_pins = 13
    env.components = [
        Component(1, 2, 0, []),
        Component(3, 2, 1, []),
        Component(5, 2, 2, []),
        Component(7, 2, 3, []),
        Component(9, 2, 4, []),
    ]
    env.net_pins = {
        0: [
            Pin(-1, -1, 0, -1, 0),
            Pin(-1, -1, 1, -1, 0),
            Pin(-1, -1, 2, -1, 0),
            Pin(-1, -1, 3, -1, 0),
            Pin(-1, -1, 4, -1, 0),
        ],
        1: [
            Pin(-1, -1, 5, -1, 1),
            Pin(-1, -1, 6, -1, 1),
            Pin(-1, -1, 7, -1, 1),
            Pin(-1, -1, 8, -1, 1),
            Pin(-1, -1, 9, -1, 1),
        ],
        2: [
            Pin(-1, -1, 10, -1, 2),
            Pin(-1, -1, 11, -1, 2),
            Pin(-1, -1, 12, -1, 2),
        ],
    }
    components_available_space = {
        0: 2,
        1: 6,
        2: 10,
        3: 14,
        4: 18,
    }

    env.num_components_w_pins = 3

    # Test for net 0
    env.allocate_pins_to_components_for_net(
        0, components_available_space, env.num_components_w_pins
    )
    # Test for net 1
    env.allocate_pins_to_components_for_net(
        1, components_available_space, env.num_components_w_pins
    )
    # Test for net 2
    env.allocate_pins_to_components_for_net(
        2, components_available_space, env.num_components_w_pins
    )

    # Check that all pins in self.net_pins have been allocated to a component
    for net in env.net_pins:
        for pin in env.net_pins[net]:
            assert pin.component_id != -1


def test_generate_instance(env_low_complexity):
    """Test the generation of an instance on a grid of size 6x6"""
    env = env_low_complexity
    for _ in range(10):
        env.generate_instances()
        components = env.components
        total_pins = 0
        for component in components:
            total_pins += len(component.pins)
        assert total_pins == env.total_num_pins


def test_update_placement_mask(env_low_complexity):
    """Test update placement mask function on an environment with grid size 6x6"""
    env = env_low_complexity
    env.generate_instances()
    pins = [Pin(0, 0, 1, 0, 0), Pin(0, 2, 2, 0, 0)]
    env.components = [
        Component(1, 2, 0, pins[0:1]),
        Component(3, 2, 1, pins[1:]),
    ]
    env.placement_mask = np.array([1, 1, 1, 1])
    env.update_placement_mask(env.components[1])

    assert np.array_equal(env.placement_mask, np.array([1, 2, 1, 1]))


def test_update_all_components_feature(env_low_complexity):
    """Test update all components feature function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]
    env.current_component = env.components[0]

    all_components_feature = env.all_components_feature.copy()
    env.update_all_components_feature(env.components)
    assert not np.array_equal(all_components_feature, env.all_components_feature)


def test_reset(env_low_complexity):
    """Test reset function on an environment with grid size 6x6, components
    with max dimension 4x4 and min dimension 2x2. Max number of components 4.
    """
    env = env_low_complexity
    env.generate_instances()
    state = env.reset()
    assert env.observation_space.contains(
        state
    ), "State after reset is not in observation space"
    assert state["grid"].shape == (6, 6)
    assert not np.all(state["grid"])

    first_comp = env.current_component
    max_id = 0

    assert np.array_equal(state["action_mask"], env.compute_action_mask(first_comp))

    for component in env.components:
        all_components_feature = state["all_components_feature"][component.comp_id, :][
            0 : len(component.calculate_feature(env.area))
        ]
        assert np.array_equal(
            all_components_feature,
            component.calculate_feature(env.area),
        )
        if max_id < component.comp_id:
            max_id = component.comp_id
    if max_id < env.max_num_components - 1:
        assert not np.all(state["all_components_feature"][max_id + 1 :, :])


def test_step(env_low_complexity):
    """Test step function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]
    env.current_component = env.components[0]
    env.placement_mask[len(env.components) :] = 0

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    # Step through environment
    state, reward, done, info = env.step((0, 0, 0))
    second_comp = env.current_component

    assert np.all(state["grid"][: env.components[0].h, : env.components[0].w])
    assert np.array_equal(state["placement_mask"], np.array([2, 3, 0, 0]))
    assert np.array_equal(state["action_mask"], env.compute_action_mask(second_comp))
    all_components_feature = state["all_components_feature"][
        env.components[0].comp_id, :
    ][0 : len(env.components[0].calculate_feature(env.area))]
    assert np.array_equal(
        all_components_feature,
        env.components[0].calculate_feature(env.area),
    )
    assert reward == 0
    assert not done
    assert info == {}


def test_validate_action(env_low_complexity):
    """Test validate action function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]

    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    assert env.validate_action(0, 0, 0)
    assert not env.validate_action(4, 5, 0)
    assert env.validate_action(2, 3, 1)
    assert not env.validate_action(5, 4, 1)


def test_rows_cols_to_mask(env_low_complexity):
    """Test rows cols to mask function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]

    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((1, 1, 0))
    rows_to_mask, cols_to_mask = env.rows_cols_to_mask(env.current_component, 0)

    assert rows_to_mask[0] == 5
    assert cols_to_mask[0] == 5
    assert len(rows_to_mask) == 2
    assert len(cols_to_mask) == 2


def test_compute_action_mask_orientation(env_low_complexity):
    """Test compute action mask orientation function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 0, 0))
    env.step((0, 1, 2))
    action_mask_orientation = env.compute_action_mask_orientation(
        Component(4, 2, 2, pins), 1
    )

    assert action_mask_orientation[1, 4] == 0
    assert action_mask_orientation[4, 1] == 1


def test_compute_action_mask(env_low_complexity):
    """Test compute action mask function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]

    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 0, 0))
    env.step((0, 2, 3))
    env.action_mask = env.compute_action_mask(Component(2, 2, 2, pins))

    assert env.action_mask[0, 2, 3] == 0
    assert env.action_mask[0, 4, 0] == 1


def test_compute_if_done(env_low_complexity):
    """Test compute if done function on an environment with grid size 6x6"""
    env = env_low_complexity

    # Reset environment and set components and current component
    env.reset()
    env.generate_instances()
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 1, 1, 0, 0)]
    env.components = [Component(2, 2, 0, pins[0:1]), Component(3, 3, 1, pins[1:])]

    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 0, 0))
    env.step((2, 3, 0))

    assert env.compute_if_done()


def test_update_all_pins_feature(env_30_30_low_complexity):
    """Test update_all_pins_feature function on an environment with grid size 30x30"""
    env = env_30_30_low_complexity
    env.reset()

    # Hard code env for testing purposes
    env.num_nets = 3
    env.components = [
        Component(1, 3, 0, [Pin(0, 0, 0, 0, 0), Pin(0, 2, 1, 0, 1)]),
        Component(4, 2, 1, [Pin(3, 1, 0, 1, 1), Pin(1, 0, 1, 1, 2)]),
        Component(5, 5, 2, [Pin(2, 0, 0, 2, 2), Pin(2, 2, 1, 2, 0)]),
    ]
    env.all_pins_num_feature = np.zeros(
        (env.max_num_components, env.max_num_pins_per_component, 4)
    )
    env.all_pins_cat_feature = np.zeros(
        (env.max_num_components, env.max_num_pins_per_component, 1)
    )
    env.update_all_pins_feature(env.components)
    # Check that the it's all 0 starting from the 4th row
    assert np.all(env.all_pins_num_feature[3:, :] == 0)
    assert np.all(env.all_pins_cat_feature[3:, :] == 0)

    env.current_component = env.components[0]
    env.action_mask = env.compute_action_mask(env.current_component)
    env.step((0, 28, 26))
    assert np.all(env.all_pins_num_feature[0, 0] == np.array([0, 0, 28, 26]))
    assert np.all(env.all_pins_cat_feature[0, 0] == 0)
    assert np.all(env.all_pins_num_feature[0, 1] == np.array([0, 2, 28, 28]))
    assert np.all(env.all_pins_cat_feature[0, 1] == 1)
    assert np.all(env.all_pins_num_feature[0, 2:] == 0)
    assert np.all(env.all_pins_cat_feature[0, 2:] == 0)

    env.step((1, 0, 0))
    assert np.all(env.all_pins_num_feature[1, 0] == np.array([1, 0, 1, 0]))
    assert np.all(env.all_pins_cat_feature[1, 0] == 1)
    assert np.all(env.all_pins_num_feature[1, 1] == np.array([0, 2, 0, 2]))
    assert np.all(env.all_pins_cat_feature[1, 1] == 2)
    assert np.all(env.all_pins_num_feature[1, 2:] == 0)
    assert np.all(env.all_pins_cat_feature[1, 2:] == 0)

    assert np.all(env.all_pins_num_feature[2, 0] == np.array([2, 0, -1, -1]))
    assert np.all(env.all_pins_cat_feature[2, 0] == 2)
    assert np.all(env.all_pins_num_feature[2, 1] == np.array([2, 2, -1, -1]))
    assert np.all(env.all_pins_cat_feature[2, 1] == 0)
    assert np.all(env.all_pins_num_feature[2, 2:] == 0)
    assert np.all(env.all_pins_cat_feature[2, 2:] == 0)
