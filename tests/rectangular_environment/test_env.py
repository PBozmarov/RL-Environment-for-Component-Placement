import numpy as np
from environment.dummy_env_rectangular import Component


def test_environment_contains(env_grid_size_6_component_size_2_4_4):
    """Test the contains method of the environment."""
    env = env_grid_size_6_component_size_2_4_4
    obs = env.reset()
    assert env.observation_space.contains(obs)


def test_action_space_sample(env_grid_size_6_6_component_size_1_1_3):
    """Test the action space sample method"""
    env = env_grid_size_6_6_component_size_1_1_3
    env.reset()
    action = env.action_space.sample()
    grid_before = env.grid.copy()
    env.step(action)
    assert not np.array_equal(grid_before, env.grid)


def test_action_space_contains(env_grid_size_6_6_component_size_1_1_3):
    """Test the action space contains method"""
    env = env_grid_size_6_6_component_size_1_1_3
    env.reset()
    assert env.action_space.contains((0, 0, 0))
    assert not env.action_space.contains((0, 6, 0))
    assert not env.action_space.contains((6, 0, 0))
    assert not env.action_space.contains((6, 6, 0))


def test_instance_generation(env_grid_size_6_component_size_2_4_4):
    """Test the instance generation method for an environment with grid
    size 6x6, minimum component dimension 2x2, maximum component dimension 4x4,
    minimum component dimension 2x2 and a maximum  of 4 components."""
    env = env_grid_size_6_component_size_2_4_4
    env.generate_instances()

    assert len(env.components) <= 4
    assert len(env.components) >= 1
    assert min([component.w for component in env.components]) >= 2
    assert max([component.w for component in env.components]) <= 4
    assert min([component.h for component in env.components]) >= 2
    assert max([component.h for component in env.components]) <= 4


def test_update_component_mask(env_fixed_components):
    """Test the update component mask method for an environment with 3
    components and max component size of 4."""
    env = env_fixed_components
    env.update_component_mask(env.components)

    assert np.array_equal([1, 1, 1, 0], env.component_mask)


def test_update_placement_mask(env_fixed_components):
    """Test the update placement mask method with environment with grid
    size 6x6 and max num components 4, where the second of two components
    is placed."""
    env = env_fixed_components
    env.update_placement_mask(env.components[1])

    assert np.array_equal(env.placement_mask, np.array([0, 1, 0, 0]))


def test_update_grid_original_orientation(env_fixed_components_reset):
    """Test the update grid method for an environment with grid
    size 6x6, where a component is to be placed in its original
    orientation."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    env.update_grid(env.components[0], 0, 0, 0)
    assert env.grid[0, 0] == 1
    assert env.grid[0, 1] == 1


def test_update_grid_rotated(env_fixed_components_reset):
    """Test the update grid method for an environment with grid
    size 6x6, where a  component is to be placed in its rotated
    orientation."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    env.update_grid(env.components[0], 1, 0, 0)
    assert env.grid[0, 0] == 1
    assert env.grid[1, 0] == 1


def test_compute_done_place_all(env_fixed_components_reset):
    """Test the compute done method for an environment with grid
    size 6x6 where all the components have been placed."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    env.step((0, 0, 0))
    env.step((0, 1, 0))
    done = env.compute_if_done()

    assert done


def test_compute_done_action_mask_full(env_fixed_components_reset_compute_done):
    """Test the compute done method for an environment with grid
    size 6x6 where there are no more possible actions."""
    env = env_fixed_components_reset_compute_done

    # Reset and set components
    env.reset()
    env.components = [
        Component(1, 2, 0),
        Component(3, 4, 1),
        Component(4, 3, 2),
        Component(4, 4, 4),
    ]
    env.current_component = env.components[0]

    # Step in environment
    env.step((1, 0, 0))
    env.step((0, 2, 2))
    done = env.compute_if_done()
    assert done


def test_reset(env_grid_size_6_component_size_2_4_4):
    """Test reset function on an environment with grid size 6x6, components
    with max dimension 4x4 and min dimension 2x2. Max.
    number of components 4."""
    env = env_grid_size_6_component_size_2_4_4
    state = env.reset()
    assert state["grid"].shape == (6, 6)
    assert not np.all(state["grid"])

    component_mask_check = [1] * (len(env.components)) + [0] * (
        env.max_num_components - len(env.components)
    )
    first_comp = env.current_component
    max_id = 0

    assert np.array_equal(state["placement_mask"], np.array([0, 0, 0, 0]))
    assert np.array_equal(state["component_mask"], np.array(component_mask_check))
    assert np.array_equal(state["action_mask"], env.compute_action_mask(first_comp))

    for component in env.components:
        assert np.array_equal(
            state["all_components_feature"][component.comp_id, :],
            component.calculate_feature(env.area),
        )
        if max_id < component.comp_id:
            max_id = component.comp_id
    if max_id < env.max_num_components - 1:
        assert not np.all(state["all_components_feature"][max_id + 1 :, :])


def test_validate_action(env_fixed_components_reset):
    """Test validate action function on an environment with grid size 6x6,
    components with max dimension 4x4 and min dimension 2x2. Max number of
    components 4."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    assert env.validate_action(0, 0, 0)
    assert not env.validate_action(0, 4, 5)
    assert env.validate_action(1, 2, 3)
    assert not env.validate_action(1, 5, 4)


def test_compute_action_mask(env_fixed_components_reset):
    """Test compute action mask function on an environment with grid size 6x6,
    components with max dimension 4x4 and min dimension 2x2. Max number of components
    4."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 0, 0))
    env.step((0, 2, 3))
    print(env.action_mask)
    env.action_mask = env.compute_action_mask(Component(2, 2, 2))

    assert env.action_mask[0, 2, 3] == 0
    assert env.action_mask[0, 4, 0] == 1


def test_compute_action_mask_orientation(env_fixed_components_reset):
    """Test compute action mask function on an environment with grid size 6x6,
    components with max dimension 4x4 and min dimension 2x2. Max number of
    components 4."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 0, 0))
    env.step((0, 1, 2))
    action_mask_orientation = env.compute_action_mask_orientation(Component(4, 2, 2), 1)

    assert action_mask_orientation[1, 4] == 0
    assert action_mask_orientation[4, 1] == 1


def test_rows_cols_to_mask(env_fixed_components_reset):
    """Test rows cols to mask function on an environment with grid size 6x6,
    components with max dimension 4x4 and min dimension 2x2. Max number of
    components 4."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    # Step through environment
    env.step((0, 1, 1))
    rows_to_mask, cols_to_mask = env.rows_cols_to_mask(env.current_component, 0)

    assert rows_to_mask[0] == 5
    assert cols_to_mask[0] == 5
    assert len(rows_to_mask) == 2
    assert len(cols_to_mask) == 1


def test_step(env_fixed_components_reset):
    """Test step function on an environment with grid size 6x6, components with
    max dimension 4x4 and min dimension 2x2. Max number of components 4. 2
    Components."""
    env = env_fixed_components_reset

    # Reset environment and set components and current component
    env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    # Step through environment
    state, reward, done, info = env.step((0, 0, 0))
    second_comp = env.current_component

    assert np.all(state["grid"][: env.components[0].h, : env.components[0].w])
    assert np.array_equal(state["placement_mask"], np.array([1, 0, 0, 0]))
    assert np.array_equal(state["component_mask"], np.array([1, 1, 0, 0]))
    assert np.array_equal(state["action_mask"], env.compute_action_mask(second_comp))
    assert np.array_equal(
        state["all_components_feature"][env.components[0].comp_id, :],
        env.components[0].calculate_feature(env.area),
    )
    assert reward == 1.0
    assert not done
    assert info == {}


def test_observation_replacement(env_fixed_components_reset_obs):
    """Test that the observation is replaced with a new one when step is called"""
    env = env_fixed_components_reset_obs

    # Reset the environment and set the components to a fixed list
    obs = env.reset()
    env.components = [Component(1, 2, 0), Component(3, 2, 1)]
    env.current_component = env.components[0]

    # Update masks
    env.action_mask = env.compute_action_mask(env.current_component)
    env.update_component_mask(env.components)
    env.update_all_components_feature(env.components)

    # Find a valid action and step through environment
    found_valid_action = False
    while not found_valid_action:
        action = env.action_space.sample()
        found_valid_action = env.validate_action(*action)
    new_obs, reward, done, info = env.step(action)
    assert not np.array_equal(obs["grid"], new_obs["grid"]), "Grid should have changed!"
    assert not np.array_equal(
        obs["action_mask"], new_obs["action_mask"]
    ), "Action mask should have changed!"
    assert not np.array_equal(
        obs["placement_mask"], new_obs["placement_mask"]
    ), "Placement mask should have changed!"

    found_valid_action = False
    while not found_valid_action:
        action = env.action_space.sample()
        found_valid_action = env.validate_action(*action)
    new_obs1, reward, done, info = env.step(action)
    assert not np.array_equal(
        new_obs["grid"], new_obs1["grid"]
    ), "Grid should have changed!"
    assert not np.array_equal(
        new_obs["action_mask"], new_obs1["action_mask"]
    ), "Action mask should have changed!"
    assert not np.array_equal(
        new_obs["placement_mask"], new_obs1["placement_mask"]
    ), "Placement mask should have changed!"
