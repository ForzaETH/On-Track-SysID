import numpy as np

from on_track_sysid.pacejka import axle_loads, pacejka_formula, slip_angles
from on_track_sysid.types import PacejkaParams, VehicleParams


def lateral_dynamics(vehicle: VehicleParams, tires: PacejkaParams, v_x, v_y, omega, delta, a_x=0.0):
    """Single-track lateral dynamics with Pacejka tires.

    Works on scalars and numpy arrays.

    Returns:
        v_y_dot, omega_dot
    """
    alpha_f, alpha_r = slip_angles(vehicle, v_x, v_y, omega, delta)
    F_zf, F_zr = axle_loads(vehicle, a_x)
    F_yf = pacejka_formula(tires.front, alpha_f, F_zf, tires.friction_coeff)
    F_yr = pacejka_formula(tires.rear, alpha_r, F_zr, tires.friction_coeff)

    v_y_dot = (F_yr + F_yf * np.cos(delta)) / vehicle.m - v_x * omega
    omega_dot = (F_yf * vehicle.lf * np.cos(delta) - F_yr * vehicle.lr) / vehicle.Iz
    return v_y_dot, omega_dot
