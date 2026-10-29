import numpy as np
from scipy.optimize import least_squares

from on_track_sysid.types import VehicleParams

G = 9.81

# (lower, upper) bounds on [B, C, D, E]. C >= 1 keeps a force peak and E <= 1 a monotonic
# rise to it (Goblirsch et al., IV 2025)
DEFAULT_BOUNDS = ([1.0, 1.0, 0.1, 0.0], [25.0, 4.0, 2.0, 1.0])
# Which of [B, C, D, E] are fitted. E only shapes the sliding region, which normal driving
# does not reach, so it keeps its start value (Goblirsch et al., IV 2025)
DEFAULT_FIT_MASK = (True, True, True, False)


def pacejka_formula(coeffs, alpha, F_z, mu=1.0):
    """Lateral tire force of the simplified Pacejka magic formula."""
    B, C, D, E = coeffs
    return mu * F_z * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))


def axle_loads(vehicle: VehicleParams, a_x=0.0):
    """Vertical front and rear axle loads including longitudinal load transfer."""
    F_zf = vehicle.m * (G * vehicle.lr - a_x * vehicle.h_cg) / vehicle.wheelbase
    F_zr = vehicle.m * (G * vehicle.lf + a_x * vehicle.h_cg) / vehicle.wheelbase
    return F_zf, F_zr


def slip_angles(vehicle: VehicleParams, v_x, v_y, omega, delta):
    alpha_f = -np.arctan((v_y + omega * vehicle.lf) / v_x) + delta
    alpha_r = -np.arctan((v_y - omega * vehicle.lr) / v_x)
    return alpha_f, alpha_r


def steady_state_forces(vehicle: VehicleParams, v_x, v_y, omega, delta, a_x=0.0):
    """Slip angles, axle loads and lateral axle forces assuming steady-state cornering.

    Returns:
        alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr
    """
    alpha_f, alpha_r = slip_angles(vehicle, v_x, v_y, omega, delta)
    F_zf, F_zr = axle_loads(vehicle, a_x)
    F_yf = vehicle.m * vehicle.lr * v_x * omega / (vehicle.wheelbase * np.cos(delta))
    F_yr = vehicle.m * vehicle.lf * v_x * omega / vehicle.wheelbase
    return alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr


def fit_pacejka(alpha, F_z, F_y, start, mu=1.0, bounds=DEFAULT_BOUNDS, fit_mask=DEFAULT_FIT_MASK):
    """Least-squares fit of [B, C, D, E] to measured (alpha, F_y) pairs.

    The start point is clipped into the bounds, as least_squares requires. Coefficients
    with fit_mask False keep their clipped start value.
    """
    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    x0 = np.clip(np.asarray(start, dtype=float), lower + 1e-6, upper - 1e-6)
    free = np.asarray(fit_mask, dtype=bool)
    F_z = np.broadcast_to(F_z, np.shape(alpha))

    def residuals(free_coeffs):
        coeffs = x0.copy()
        coeffs[free] = free_coeffs
        return pacejka_formula(coeffs, alpha, F_z, mu) - F_y

    coeffs = x0.copy()
    # dogbox: trf stalls in a local minimum on the bounds for typical starts
    coeffs[free] = least_squares(residuals, x0[free], bounds=(lower[free], upper[free]), method='dogbox').x
    return coeffs
