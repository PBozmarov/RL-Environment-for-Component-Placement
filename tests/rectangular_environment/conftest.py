import pytest
from environment.dummy_env_rectangular import Component, DummyPlacementEnv


@pytest.fixture()
def component_size_3_4_id_2():
    """Returns a component with height 3 and width 4."""
    return Component(3, 4, 2)


@pytest.fixture()
def env_grid_size_6_component_size_2_4_4():
    """Returns an environment with grid size 6x6, minimum component dimension 2x2,
    maximum component dimension 4x4, minimum component dimension 2x2 and
    a maximum  of 4 components."""
    env = DummyPlacementEnv(6, 6, 2, 4, 2, 4, 4, 1)
    return env


@pytest.fixture()
def env_fixed_components():
    """Returns an environment with a fixed list of components of length 3 each
    with different dimensions."""
    env = DummyPlacementEnv(6, 6, 2, 4, 2, 4, 4, 1)
    component_list = [
        Component(2, 2, 0),
        Component(2, 3, 1),
        Component(3, 2, 2),
    ]
    env.components = component_list
    return env


@pytest.fixture()
def env_fixed_components_big():
    """Returns an environment with a fixed list of components of length 3 each
    with different dimensions."""
    env = DummyPlacementEnv(16, 16, 2, 2, 2, 2, 4, 1)
    component_list = [
        Component(2, 2, 0),
        Component(2, 3, 1),
        Component(3, 2, 2),
    ]
    env.components = component_list
    return env


@pytest.fixture()
def env_fixed_components_reset():
    """Returns an environment with a fixed list of components of length 3 each
    with different dimensions."""
    env = DummyPlacementEnv(6, 6, 2, 4, 2, 4, 4, 1)
    return env


@pytest.fixture()
def env_fixed_components_reset_obs():
    """Returns an environment with a fixed list of components of length 3 each
    with different dimensions."""
    env = DummyPlacementEnv(6, 6, 2, 4, 2, 4, 4, 1)
    return env


@pytest.fixture()
def env_fixed_components_reset_compute_done():
    """Returns an environment with a fixed list of components of length 3 each
    with different dimensions."""
    env = DummyPlacementEnv(6, 6, 2, 4, 2, 4, 4, 1)
    return env


@pytest.fixture()
def env_grid_size_6_6_component_size_1_1_3():
    """Returns an environment with grid size 6x6, minimum component dimension 1x1,
    maximum component dimension 1x1 and a maximum  of 3 components."""
    env = DummyPlacementEnv(6, 6, 1, 1, 1, 1, 3, 1)
    return env
