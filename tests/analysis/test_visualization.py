"""plot_timeseries / plot_xy_trajectory on the recorded flight, and the square XY axes."""
import numpy as np
import pytest

from cf_lab.analysis import load_dataset, plot_timeseries, plot_xy_trajectory
from cf_lab.analysis.visualization import _square_xy_axes


@pytest.fixture(scope="module")
def flight_data(flight_run):
    return load_dataset(flight_run, include_mocap_position=True, include_setpoints=True)


@pytest.mark.parametrize("xs, ys", [
    ([0.0, 3.0], [0.0, 0.2]),  # wide pass
    ([-0.05, 0.05], [-2.0, 1.5]),  # tall pass
    ([0.31, 0.31], [-0.7, -0.7]),  # hover on one spot
    ([-1.23, -0.98, -1.11], [4.4, 4.9, 4.6]),
])
def test_square_xy_axes_have_equal_spans_and_ticks_that_enclose_the_data(xs, ys):
    x_range, y_range, x_ticks, y_ticks = _square_xy_axes(xs, ys)

    assert x_range[1] - x_range[0] == pytest.approx(y_range[1] - y_range[0])
    assert len(x_ticks) == len(y_ticks)
    step = np.diff(x_ticks)
    assert np.allclose(step, step[0]) and np.allclose(np.diff(y_ticks), step[0])
    assert x_ticks[0] <= min(xs) and max(xs) <= x_ticks[-1]
    assert y_ticks[0] <= min(ys) and max(ys) <= y_ticks[-1]


def test_plot_xy_trajectory_draws_every_recorded_pair_on_square_axes(flight_data):
    figure = plot_xy_trajectory(flight_data)

    traces = {trace.name: trace for trace in figure.data}
    assert set(traces) == {"EKF", "Zeroed mocap", "Setpoint"}
    for name, (x, y) in {"EKF": ("x", "y"), "Zeroed mocap": ("mocap_x", "mocap_y"),
                         "Setpoint": ("setpoint_x", "setpoint_y")}.items():
        assert traces[name].x == flight_data[x] and traces[name].y == flight_data[y]
    assert traces["Setpoint"].line.shape == "hv"
    assert figure.layout.yaxis.scaleanchor == "x"


def test_plot_timeseries_puts_each_group_in_its_own_row_with_its_shared_unit(flight_data):
    figure = plot_timeseries(flight_data, {"Position": ("x", "mocap_x"), "Mixed": ("z", "roll")})

    rows = [(trace.name, trace.yaxis) for trace in figure.data]
    assert rows == [("x", "y"), ("mocap_x", "y"), ("z", "y2"), ("roll", "y2")]
    assert figure.data[3].y == flight_data["roll"]
    assert figure.layout.yaxis.title.text == "m"
    assert figure.layout.yaxis2.title.text is None
    with pytest.raises(KeyError, match=r"missing signal\(s\) \['thrust'\]"):
        plot_timeseries(flight_data, {"Missing": ("thrust",)})
