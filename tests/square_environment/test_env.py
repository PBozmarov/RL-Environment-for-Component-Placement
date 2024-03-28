""" Test the environment for the square components of a fixed size. """

import numpy as np


def test_compute_if_done(env_grid_size_4_size_2):
    """Example where the episode is not done and then the episode is done
    following more actions."""
    env = env_grid_size_4_size_2
    env.reset()

    action = (0, 0)
    env.step(action, verbose=True)

    # Episode not done
    is_done = env.compute_if_done()
    assert not is_done

    action = (0, 2)
    env.step(action, verbose=True)
    action = (2, 1)
    env.step(action, verbose=True)

    # Episode done
    is_done = env.compute_if_done()
    assert is_done


def test_update_grid(env_grid_size_4_size_2):
    """Test that the grid is updated with 2 actions."""
    env = env_grid_size_4_size_2
    env.reset()
    action = (0, 0)
    env.step(action)
    expected_grid = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert (env.grid == expected_grid).all()

    action = (2, 2)
    env.step(action)
    expected_grid = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert (env.grid == expected_grid).all()


def test_validate_action_overlap(env_grid_size_4_size_2):
    """Test the validate action function with overlapping components."""
    env = env_grid_size_4_size_2
    env.reset()
    action = (0, 0)
    env.step(action)

    # Component overlapping, Output=False
    action = (1, 1)
    assert not env.validate_action(action[0], action[1])


def test_validate_action_out_of_bounds(env_grid_size_4_size_2):
    """Test the validate action function with components out of bounds."""
    env = env_grid_size_4_size_2
    env.reset()
    action = (0, 0)
    env.step(action)

    # Component out of bounds, Output=False
    action = (4, 4)
    assert not env.validate_action(action[0], action[1])


def test_validate_action_correct(env_grid_size_4_size_2):
    """Test the validate action function with components placed correctly."""
    env = env_grid_size_4_size_2
    env.reset()
    action = (0, 0)
    env.step(action)

    # Component placed correctly, Output=True
    action = (2, 2)
    assert env.validate_action(action[0], action[1])


def test_update_action_mask_origin(env_grid_size_5_size_2):
    """Test the update action mask function when action is at the origin
    of the grid, i.e. (0,0)."""
    # Grid is 5x5, component size is 2
    env = env_grid_size_5_size_2
    env.reset()
    expected_grid_before_update = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    assert (env.action_mask == expected_grid_before_update).all()
    env.update_action_mask(0, 0)
    expected_grid_after_update = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert (env.action_mask == expected_grid_after_update).all()


def test_update_action_mask_left_border(env_grid_size_5_size_2):
    """Test the update action mask function when action is at the left
    border of the grid, i.e. y=0."""
    # Grid is 5x5, component size is 2
    env = env_grid_size_5_size_2
    env.reset()

    env.update_action_mask(1, 0)
    expected_grid_after_update = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert (env.action_mask == expected_grid_after_update).all()


def test_update_action_mask_top_border(env_grid_size_5_size_2):
    """Test the update action mask function when action is at the top
    border of the grid, i.e. x=0."""
    # Grid is 5x5, component size is 2
    env = env_grid_size_5_size_2
    env.reset()

    env.update_action_mask(0, 2)
    expected_grid_after_update = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert (env.action_mask == expected_grid_after_update).all()


def test_update_action_mask_top_left_corner(env_grid_size_11_10_size_3):
    """Test the update action mask function when action is at the top left corner
    of the grid, i.e., it's not possible to fit another component above or to
    the left of the component with the current action. An example
    can be: action is (1,1) and component size is 3, grid is 11x10."""
    # Grid is 11x10, component size is 3
    env = env_grid_size_11_10_size_3
    env.reset()
    expected_grid_before_update = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    assert (env.action_mask == expected_grid_before_update).all()

    env.update_action_mask(1, 1)
    expected_grid_after_update = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert (env.action_mask == expected_grid_after_update).all()


def test_update_action_mask_middle(env_grid_size_11_10_size_3):
    """Test the update action mask function when action is in the middle of the
    grid, i.e., it's possible to fit another component above, below,
    to the left, and to the right of the component with the
    current action. An example can be: action is (3,3),
    component size is 3, grid is 11x10."""
    # grid is 11x10, component size is 3
    env = env_grid_size_11_10_size_3
    env.reset()

    env.update_action_mask(3, 3)
    expected_grid_after_update = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert (env.action_mask == expected_grid_after_update).all()


def test_env_step_valid_actions(env_grid_size_4_size_2):
    """Test the step function with a sequence of valid actions."""
    env = env_grid_size_4_size_2
    env.reset()
    actions = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for action in actions:
        obs, reward, done, info = env.step(action)
    assert done
    assert env.observation_space.contains(obs)
    assert obs["grid"].sum() == 4 * 4
    assert obs["action_mask"].sum() == 0


def test_env_step_invalid_actions(env_grid_size_4_size_2):
    """Test the step function with a sequence of actions where some are
    invalid."""
    env = env_grid_size_4_size_2
    env.reset()
    actions = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)]
    err = 0
    try:
        for action in actions:
            obs, reward, done, info = env.step(action)
    except Exception:
        err = 1
    assert err == 0


def test_env_reset(env_grid_size_4_size_2):
    """Test the reset function of the environment."""
    env = env_grid_size_4_size_2
    obs = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["grid"].sum() == 0
    assert obs["action_mask"].sum() == 3 * 3


def test_observation_replacement(env_grid_size_11_10_size_3):
    """Test that the observation is replaced with a new one when step is called"""
    env = env_grid_size_11_10_size_3
    obs = env.reset()
    found_valid_action = False
    while not found_valid_action:
        action = env.action_space.sample()
        found_valid_action = env.validate_action(*action)
    new_obs, reward, done, info = env.step(action)
    assert not np.array_equal(obs["grid"], new_obs["grid"]), "Grid should have changed!"
    assert not np.array_equal(
        obs["action_mask"], new_obs["action_mask"]
    ), "Action mask should have changed!"

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
