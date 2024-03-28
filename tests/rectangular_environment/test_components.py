def test_component_init(component_size_3_4_id_2):
    """Test the component initialization."""
    component = component_size_3_4_id_2
    assert component.h == 3
    assert component.w == 4
    assert component.area == 12
    assert component.comp_id == 2
    assert not component.placed
    assert component.position == (-1, -1)


def test_component_place_component(component_size_3_4_id_2):
    """Test the component place component function."""
    component = component_size_3_4_id_2
    component.place_component(1, 2)
    assert component.placed
    assert component.position == (1, 2)


def test_component_area_ratio(component_size_3_4_id_2):
    """Test the component area ratio function which returns the ratio of the
    component area to the total area of the grid."""
    component = component_size_3_4_id_2
    assert component.area_ratio(24) == 0.5


def test_calculate_feature_not_placed(component_size_3_4_id_2):
    """Test the calculate feature method for a component that has not been
    placed yet."""
    component = component_size_3_4_id_2
    feature = component.calculate_feature(24)
    assert feature[0] == 3
    assert feature[1] == 4
    assert feature[2] == -1
    assert feature[3] == -1
    assert feature[4] == 0.5


def test_calculate_feature_placed(component_size_3_4_id_2):
    """Test the calculate feature method for a component that has been
    placed."""
    component = component_size_3_4_id_2
    component.place_component(1, 2)
    feature = component.calculate_feature(24)

    assert feature[0] == 3
    assert feature[1] == 4
    assert feature[2] == 1
    assert feature[3] == 2
    assert feature[4] == 0.5
