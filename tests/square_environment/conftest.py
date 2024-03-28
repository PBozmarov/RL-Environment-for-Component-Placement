import pytest
from environment.dummy_env_square import DummyPlacementEnv


@pytest.fixture()
def env_grid_size_4_size_2():
    """Returns an environment with grid size 4x4 and component size 2x2."""
    env = DummyPlacementEnv(4, 4, 2)
    return env


@pytest.fixture()
def env_grid_size_5_size_2():
    """Returns an environment with grid size 5x5 and component size 2x2"""
    env = DummyPlacementEnv(5, 5, 2)
    return env


@pytest.fixture()
def env_grid_size_11_10_size_3():
    """Returns an environment with grid size 11x10 and component size 3x3"""
    env = DummyPlacementEnv(11, 10, 3)
    return env
