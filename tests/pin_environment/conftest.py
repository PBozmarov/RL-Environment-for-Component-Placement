import pytest  # type: ignore
from environment.dummy_env_rectangular_pin import (
    Component,
    Pin,
    DummyPlacementEnv as DummyPlacementEnvRectPin,
)


@pytest.fixture()
def component_3x3_2_pins():
    """Returns a 3x3 component with 2 pins."""
    pin_0 = Pin(relative_x=0, relative_y=0, pin_id=0, component_id=0, net_id=0)
    pin_1 = Pin(relative_x=0, relative_y=1, pin_id=0, component_id=0, net_id=0)

    # Add absolute positions of pins
    pin_0.absolute_x, pin_0.absolute_y = 0, 0
    pin_1.absolute_x, pin_1.absolute_y = 0, 1
    return Component(h=3, w=3, comp_id=0, pins=[pin_0, pin_1])


@pytest.fixture()
def component_3x3_3_pins():
    """Returns a 3x3 component with 3 pins."""
    pin_1 = Pin(relative_x=2, relative_y=2, pin_id=0, component_id=1, net_id=1)
    pin_2 = Pin(relative_x=3, relative_y=3, pin_id=0, component_id=1, net_id=1)
    pin_3 = Pin(relative_x=4, relative_y=4, pin_id=0, component_id=1, net_id=1)

    # Add absolute positions of pins
    pin_1.absolute_x, pin_1.absolute_y = 2, 2
    pin_2.absolute_x, pin_2.absolute_y = 3, 3
    pin_3.absolute_x, pin_3.absolute_y = 4, 4
    return Component(h=3, w=3, comp_id=1, pins=[pin_1, pin_2, pin_3])


@pytest.fixture()
def component_3x3_2_pins_beam():
    """Returns a 3x3 component with 2 pins."""
    pin_0 = Pin(relative_x=0, relative_y=0, pin_id=0, component_id=0, net_id=0)
    pin_1 = Pin(relative_x=0, relative_y=1, pin_id=0, component_id=0, net_id=0)

    # Add absolute positions of pins
    pin_0.absolute_x, pin_0.absolute_y = 3, 3
    pin_1.absolute_x, pin_1.absolute_y = 3, 4

    return Component(h=3, w=3, comp_id=0, pins=[pin_0, pin_1])


@pytest.fixture()
def component_3x3_5_pins_beam():
    """Returns a 3x3 component with 5 pins."""
    pin_1 = Pin(relative_x=0, relative_y=0, pin_id=0, component_id=1, net_id=1)
    pin_2 = Pin(relative_x=0, relative_y=1, pin_id=0, component_id=1, net_id=1)
    pin_3 = Pin(relative_x=1, relative_y=0, pin_id=0, component_id=1, net_id=1)
    pin_4 = Pin(relative_x=1, relative_y=1, pin_id=0, component_id=1, net_id=1)
    pin_5 = Pin(relative_x=2, relative_y=2, pin_id=0, component_id=1, net_id=1)

    # Add absolute positions of pins
    pin_1.absolute_x, pin_1.absolute_y = 0, 0
    pin_2.absolute_x, pin_2.absolute_y = 0, 1
    pin_3.absolute_x, pin_3.absolute_y = 1, 0
    pin_4.absolute_x, pin_4.absolute_y = 1, 1
    pin_5.absolute_x, pin_5.absolute_y = 2, 2

    return Component(h=3, w=3, comp_id=1, pins=[pin_1, pin_2, pin_3, pin_4, pin_5])


@pytest.fixture()
def component_3x3_2_pins_reward_1():
    """Returns a 3x3 component with 2 pins."""
    pin_1 = Pin(relative_x=0, relative_y=2, pin_id=0, component_id=1, net_id=1)
    pin_2 = Pin(relative_x=2, relative_y=0, pin_id=0, component_id=1, net_id=2)

    # Add absolute positions of pins
    pin_1.absolute_x, pin_1.absolute_y = 0, 2
    pin_2.absolute_x, pin_2.absolute_y = 2, 0

    return Component(h=3, w=3, comp_id=1, pins=[pin_1, pin_2])


@pytest.fixture()
def component_3x3_2_pins_reward_2():
    """Returns a 3x3 component with 2 pins."""
    pin_1 = Pin(relative_x=0, relative_y=2, pin_id=0, component_id=2, net_id=2)
    pin_2 = Pin(relative_x=2, relative_y=0, pin_id=0, component_id=2, net_id=1)

    # Add absolute positions of pins
    pin_1.absolute_x, pin_1.absolute_y = 3, 4
    pin_2.absolute_x, pin_2.absolute_y = 5, 3

    return Component(h=3, w=3, comp_id=2, pins=[pin_1, pin_2])


@pytest.fixture()
def component_2x1_1_pin_reward_1():
    """Returns a 2x1 component with 1 pin."""
    pin_1 = Pin(relative_x=0, relative_y=0, pin_id=0, component_id=3, net_id=2)
    pin_1.absolute_x, pin_1.absolute_y = 4, 1

    return Component(h=2, w=1, comp_id=3, pins=[pin_1])


@pytest.fixture()
def component_2x1_1_pin_reward_2():
    """Returns a 2x1 component with 1 pin."""
    pin_1 = Pin(relative_x=0, relative_y=0, pin_id=0, component_id=4, net_id=2)
    pin_1.absolute_x, pin_1.absolute_y = 7, 5

    return Component(h=2, w=1, comp_id=4, pins=[pin_1])


@pytest.fixture()
def component_2x2_1_pin_reward():
    """Returns a 2x2 component with 1 pin."""
    pin_1 = Pin(relative_x=0, relative_y=1, pin_id=0, component_id=5, net_id=1)
    pin_1.absolute_x, pin_1.absolute_y = 8, 1

    return Component(h=2, w=2, comp_id=5, pins=[pin_1])


@pytest.fixture()
def pin_simple_test():
    """Returns a pin with a simple test case."""
    return Pin(0, 0, 0, 0, 0)


@pytest.fixture()
def env_fixed_components_reset():
    """Returns an 6x6 grid environment to use for testing
    the reset function."""
    env = DummyPlacementEnvRectPin(
        height=6,
        width=6,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=4,
        min_num_components=2,
        min_num_nets=4,
        max_num_nets=4,
        max_num_pins_per_net=2,
    )
    return env


@pytest.fixture()
def env_upper_bound_intersections():
    """Returns an 6x6 grid environment to use for testing
    the upper bound on the number of intersections for routing."""
    env = DummyPlacementEnvRectPin(
        height=6,
        width=6,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=4,
        min_num_components=1,
        min_num_nets=2,
        max_num_nets=3,
        max_num_pins_per_net=4,
    )
    return env


@pytest.fixture()
def env_low_complexity():
    """Returns an 6x6 grid environment with
    low complexity to use for testing."""
    env = DummyPlacementEnvRectPin(
        height=6,
        width=6,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=4,
        min_num_components=2,
        min_num_nets=4,
        max_num_nets=4,
        max_num_pins_per_net=2,
    )
    return env


@pytest.fixture()
def env_10_10_low_complexity():
    """Returns an 10x10 grid environment with
    low complexity to use for testing."""
    env = DummyPlacementEnvRectPin(
        height=10,
        width=10,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=6,
        min_num_components=1,
        min_num_nets=2,
        max_num_nets=4,
        max_num_pins_per_net=5,
    )
    return env


@pytest.fixture()
def env_30_30_low_complexity():
    """Returns an 30x30 grid environment with low complexity
    to use for testing."""
    env = DummyPlacementEnvRectPin(
        height=30,
        width=30,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=5,
        min_component_h=2,
        max_component_h=5,
        max_num_components=6,
        min_num_components=1,
        min_num_nets=2,
        max_num_nets=4,
        max_num_pins_per_net=5,
        min_num_pins_per_net=2,
    )
    return env


@pytest.fixture()
def component_simple():
    """Returns a simple component with 2 pins."""
    pins = [Pin(0, 0, 0, 0, 0), Pin(0, 2, 1, 0, 0)]
    return Component(4, 3, 0, pins)


@pytest.fixture()
def env_10_10():
    """Returns a 10x10 grid environment."""
    env = DummyPlacementEnvRectPin(
        height=10,
        width=10,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=4,
        min_num_components=2,
        min_num_nets=4,
        max_num_nets=4,
        max_num_pins_per_net=2,
    )
    return env


@pytest.fixture()
def env_10_10_reward():
    """Returns a 10x10 grid environment to test reward functions."""
    env = DummyPlacementEnvRectPin(
        height=10,
        width=10,
        net_distribution=1,
        pin_spread=1,
        min_component_w=2,
        max_component_w=4,
        min_component_h=2,
        max_component_h=4,
        max_num_components=5,
        min_num_components=2,
        min_num_nets=4,
        max_num_nets=4,
        max_num_pins_per_net=2,
    )
    return env
