import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from on_track_sysid.pacejka import DEFAULT_BOUNDS, fit_pacejka, pacejka_formula, steady_state_forces
from on_track_sysid.preprocess import DELTA, OMEGA, V_X, V_Y
from on_track_sysid.types import PacejkaParams, VehicleParams


def fit_axles(tires: PacejkaParams, alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr,
              bounds=DEFAULT_BOUNDS, equal_tires: bool = False) -> PacejkaParams:
    """Fits front and rear Pacejka coefficients, starting from `tires`."""
    mu = tires.friction_coeff
    F_zf = np.broadcast_to(F_zf, np.shape(alpha_f))
    F_zr = np.broadcast_to(F_zr, np.shape(alpha_r))
    if equal_tires:
        start = (np.asarray(tires.front) + np.asarray(tires.rear)) / 2
        coeffs = fit_pacejka(np.concatenate((alpha_f, alpha_r)), np.concatenate((F_zf, F_zr)),
                             np.concatenate((F_yf, F_yr)), start, mu, bounds)
        return tires.with_coeffs(coeffs, coeffs)
    front = fit_pacejka(alpha_f, F_zf, F_yf, tires.front, mu, bounds)
    rear = fit_pacejka(alpha_r, F_zr, F_yr, tires.rear, mu, bounds)
    return tires.with_coeffs(front, rear)


def identify_steady_state(data, vehicle: VehicleParams, init: PacejkaParams, a_x=None,
                          vy_at_rear_axle: bool = True, min_speed: float = 1.0,
                          outlier_iterations: int = 3, outlier_threshold: float = 20.0,
                          bounds=DEFAULT_BOUNDS, equal_tires: bool = False) -> PacejkaParams:
    """Identifies Pacejka coefficients from recorded quasi-steady-state cornering
    (e.g. constant speed with a slow steering ramp).

    Args:
        data: (N, 4) array of v_x, v_y, omega, delta samples in time order.
        a_x: optional (N,) longitudinal acceleration, used for load transfer. Without it
            the static axle loads are used.
        vy_at_rear_axle: v_y is measured at the rear axle and is moved to the center of gravity.
        outlier_iterations: refits after discarding samples whose force error exceeds
            outlier_threshold * 2^-(i+1) [N] on either axle, for refit i = 0..outlier_iterations-1.
    """
    data = np.asarray(data, dtype=float)
    keep = data[:, V_X] > min_speed
    data = data[keep]

    v_x = median_filter(data[:, V_X], size=5)
    v_y = median_filter(data[:, V_Y], size=5)
    omega = data[:, OMEGA]
    delta = data[:, DELTA]
    if vy_at_rear_axle:
        v_y = v_y + vehicle.lr * omega
    if a_x is not None:
        a_x = sosfiltfilt(butter(2, 0.05, output='sos'), np.asarray(a_x, dtype=float)[keep])
    else:
        a_x = 0.0

    alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr = steady_state_forces(vehicle, v_x, v_y, omega, delta, a_x)
    F_zf = np.broadcast_to(F_zf, alpha_f.shape)
    F_zr = np.broadcast_to(F_zr, alpha_r.shape)

    mask = (np.abs(alpha_f) <= 2) & (np.abs(alpha_r) <= 2)
    tires = init
    for i in range(outlier_iterations + 1):
        tires = fit_axles(tires, alpha_f[mask], alpha_r[mask], F_zf[mask], F_zr[mask], F_yf[mask], F_yr[mask],
                          bounds, equal_tires)
        threshold = outlier_threshold * 2 ** -(i + 1)
        err_f = np.abs(pacejka_formula(tires.front, alpha_f, F_zf, tires.friction_coeff) - F_yf)
        err_r = np.abs(pacejka_formula(tires.rear, alpha_r, F_zr, tires.friction_coeff) - F_yr)
        mask &= (err_f <= threshold) & (err_r <= threshold)
    return tires
