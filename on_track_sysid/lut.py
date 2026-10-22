import numpy as np
from scipy.integrate import odeint

from on_track_sysid.dynamics import lateral_dynamics
from on_track_sysid.types import PacejkaParams, VehicleParams


def default_steers():
    """Fine steps up to 0.1 rad, coarse steps up to 0.4 rad."""
    fine = np.linspace(0.0, 0.1, int(0.1 / 0.0033), endpoint=False)
    coarse = np.linspace(0.1, 0.4, int(0.3 / 0.01))
    return np.concatenate((fine, coarse))


def default_vels():
    return np.linspace(0.5, 7.0, int(6.5 / 0.1))


def generate_lut(vehicle: VehicleParams, tires: PacejkaParams, steers=None, vels=None,
                 duration: float = 2.0, dt: float = 0.01):
    """Steady-state lateral acceleration for each (steering angle, velocity) pair.

    Returns:
        (len(steers) + 1, len(vels) + 1) table: row 0 holds the velocities, column 0 the
        steering angles, cells the lateral acceleration [m/s^2]. Cells without a steady
        state, or past the first maximum of a velocity column, are NaN.

    Raises:
        ValueError: if steers are negative or not increasing (the table covers one side;
            lookups mirror it by sign), or if duration / dt gives fewer than 15 samples,
            the window of the steady-state check.
    """
    steers = default_steers() if steers is None else np.asarray(steers)
    vels = default_vels() if vels is None else np.asarray(vels)
    if np.any(steers < 0) or np.any(np.diff(steers) <= 0):
        raise ValueError("steers must be nonnegative and strictly increasing")
    t = np.arange(0, duration, dt)
    if len(t) < 15:
        raise ValueError(f"duration / dt gives {len(t)} samples, the steady-state check needs at least 15")

    table = np.full((len(steers) + 1, len(vels) + 1), np.nan)
    table[0, 0] = 0.0
    table[0, 1:] = vels
    table[1:, 0] = steers
    for i, steer in enumerate(steers):
        for j, vel in enumerate(vels):
            sol = odeint(lambda x, _: lateral_dynamics(vehicle, tires, vel, x[0], x[1], steer), [0.0, 0.0], t)
            omega = sol[:, 1]
            if not np.allclose(omega[-11:-1], omega[-15:-5], rtol=1e-3):
                break  # no steady state; higher velocities will not reach one either
            table[i + 1, j + 1] = omega[-1] * vel

    _keep_below_first_max(table)
    return table


def _keep_below_first_max(table):
    """Several steering angles must not map to the same lateral acceleration:
    keep each velocity column only up to its first maximum."""
    for j in range(1, table.shape[1]):
        a_lat = table[1:, j]
        a_lat = a_lat[~np.isnan(a_lat)]
        if a_lat.size == 0:
            continue
        drops = np.flatnonzero(np.diff(a_lat) < 0)
        first_max = drops[0] if drops.size else a_lat.size - 1
        table[first_max + 2:, j] = np.nan
