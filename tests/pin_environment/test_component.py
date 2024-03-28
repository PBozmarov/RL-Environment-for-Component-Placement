def test_place_component_orientation_0(component_simple):
    """Test placing a component with orientation 0 degrees."""
    component_simple.place_component(0, 2, 0)
    assert component_simple.pins[0].absolute_x == 2
    assert component_simple.pins[0].absolute_y == 0
    assert component_simple.pins[1].absolute_x == 2
    assert component_simple.pins[1].absolute_y == 2


def test_place_component_orientation_90(component_simple):
    """Test placing a component with orientation 90 degrees."""
    component_simple.place_component(1, 0, 0)
    assert component_simple.pins[0].absolute_x == 0
    assert component_simple.pins[0].absolute_y == 3
    assert component_simple.pins[1].absolute_x == 2
    assert component_simple.pins[1].absolute_y == 3


def test_place_component_orientation_180(component_simple):
    """Test placing a component with orientation 180 degrees."""
    component_simple.place_component(2, 0, 0)
    assert component_simple.pins[0].absolute_x == 3
    assert component_simple.pins[0].absolute_y == 2
    assert component_simple.pins[1].absolute_x == 3
    assert component_simple.pins[1].absolute_y == 0


def test_place_component_orientation_270(component_simple):
    """Test placing a component with orientation 270 degrees."""
    component_simple.place_component(3, 0, 0)
    assert component_simple.pins[0].absolute_x == 2
    assert component_simple.pins[0].absolute_y == 0
    assert component_simple.pins[1].absolute_x == 0
    assert component_simple.pins[1].absolute_y == 0
