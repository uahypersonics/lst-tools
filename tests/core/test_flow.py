import numpy as np
import pytest
from lst_tools.core.grid import Grid
from lst_tools.core.flow import Flow


class TestFlow:
    def test_flow_creation(self):
        """Test creating a Flow with valid parameters"""
        # Create a grid
        x = np.array([[0, 1], [0, 1]])
        y = np.array([[0, 0], [1, 1]])
        grid = Grid(x=x, y=y)

        # Create flow fields
        velocity_u = np.array([[1.0, 1.0], [1.0, 1.0]])
        velocity_v = np.array([[0.0, 0.0], [0.0, 0.0]])
        pressure = np.array([[101325.0, 101325.0], [101325.0, 101325.0]])

        fields = {
            "velocity_u": velocity_u,
            "velocity_v": velocity_v,
            "pressure": pressure,
        }

        attrs = {"density": 1.225, "temperature": 288.15}

        flow = Flow(grid=grid, fields=fields, attrs=attrs)

        assert flow.grid == grid
        assert flow.fields == fields
        assert flow.attrs == attrs

    def test_flow_field_method(self):
        """Test the field method to retrieve flow fields"""
        # Create a grid
        x = np.array([[0, 1], [0, 1]])
        y = np.array([[0, 0], [1, 1]])
        grid = Grid(x=x, y=y)

        # Create flow fields
        velocity_u = np.array([[1.0, 1.0], [1.0, 1.0]])
        velocity_v = np.array([[0.0, 0.0], [0.0, 0.0]])

        fields = {"velocity_u": velocity_u, "velocity_v": velocity_v}

        attrs = {"density": 1.225}

        flow = Flow(grid=grid, fields=fields, attrs=attrs)

        # Test retrieving fields
        assert np.array_equal(flow.field("velocity_u"), velocity_u)
        assert np.array_equal(flow.field("velocity_v"), velocity_v)

    def test_flow_field_method_shape_mismatch(self):
        """Test that field method raises ValueError for shape mismatch"""
        # Create a grid
        x = np.array([[0, 1], [0, 1]])
        y = np.array([[0, 0], [1, 1]])
        grid = Grid(x=x, y=y)  # Shape (2, 2)

        # Create a field with wrong shape
        # Shape (2,) instead of (2, 2)
        wrong_shape_field = np.array([1.0, 1.0])

        fields = {"wrong_field": wrong_shape_field}

        attrs = {"density": 1.225}

        flow = Flow(grid=grid, fields=fields, attrs=attrs)

        # Should raise ValueError due to shape mismatch
        with pytest.raises(
            ValueError, match="field 'wrong_field' shape .* != grid shape .*"
        ):
            flow.field("wrong_field")

    def test_flow_immutable(self):
        """Test that Flow is immutable (frozen)"""
        # Create a grid
        x = np.array([0, 1])
        y = np.array([0, 1])
        grid = Grid(x=x, y=y)

        # Create flow fields
        fields = {"velocity_u": np.array([1.0, 1.0])}
        attrs = {"density": 1.225}

        flow = Flow(grid=grid, fields=fields, attrs=attrs)

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            flow.grid = Grid(x=np.array([2, 3]), y=np.array([2, 3]))

        with pytest.raises(AttributeError):
            flow.fields = {"new_field": np.array([1.0])}

        with pytest.raises(AttributeError):
            flow.attrs = {"new_attr": "value"}

    def test_flow_empty_fields(self):
        """Test creating a Flow with empty fields"""
        x = np.array([0, 1])
        y = np.array([0, 1])
        grid = Grid(x=x, y=y)

        flow = Flow(grid=grid, fields={}, attrs={})

        assert flow.grid == grid
        assert flow.fields == {}
        assert flow.attrs == {}

    def test_flow_field_not_found(self):
        """Test that field method raises KeyError for non-existent field"""
        x = np.array([0, 1])
        y = np.array([0, 1])
        grid = Grid(x=x, y=y)

        fields = {"velocity_u": np.array([1.0, 1.0])}
        attrs = {"density": 1.225}

        flow = Flow(grid=grid, fields=fields, attrs=attrs)

        with pytest.raises(KeyError):
            flow.field("non_existent_field")
