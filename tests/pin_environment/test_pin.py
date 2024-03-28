def test_pin_update_position(pin_simple_test):
    """Test whether the pin position is updated correctly."""
    pin = pin_simple_test
    pin.update_pin_position(0, 0)
    assert pin.absolute_x == 0
    assert pin.absolute_y == 0


def test_get_net_pin_positions(env_10_10, component_3x3_2_pins, component_3x3_3_pins):
    """Test whether the correct pin positions are returned for a
    given environment."""
    env = env_10_10
    component1 = component_3x3_2_pins
    component2 = component_3x3_3_pins

    # Create dictionary for the nets and their pins
    net_pins = {}
    net_pins["net1"] = component1.pins
    net_pins["net2"] = component2.pins
    env.net_pins = net_pins
    assert env.get_net_pin_positions() == {
        "net1": [(0, 0), (0, 1)],
        "net2": [(2, 2), (3, 3), (4, 4)],
    }
