import numpy as np
from scipy.signal import butter, filtfilt

# Column order of every data array: v_x, v_y, omega, delta
V_X, V_Y, OMEGA, DELTA = range(4)


def filter_data(data, lr: float, vy_at_rear_axle: bool = True, steer_delay_steps: int = 5):
    """Low-pass filters all columns, delays the steering by `steer_delay_steps` samples
    and moves v_y from the rear axle to the center of gravity when needed."""
    b, a = butter(N=3, Wn=0.1, btype='low')
    data = filtfilt(b, a, data, axis=0)
    data[:, DELTA] = np.roll(data[:, DELTA], steer_delay_steps)
    data = data[steer_delay_steps:]
    if vy_at_rear_axle:
        data[:, V_Y] += lr * data[:, OMEGA]
    return data


def mirror_data(data):
    """Doubles the data set with its left-right mirror image (v_y, omega and delta negated)."""
    mirrored = data.copy()
    mirrored[:, [V_Y, OMEGA, DELTA]] *= -1
    return np.concatenate((data, mirrored))


def split_segments(data, segment_starts, min_len: int):
    """Splits data at the samples where a time-contiguous segment starts and drops segments
    shorter than min_len samples. Without segment_starts the data is one segment."""
    if segment_starts is None:
        return [data] if len(data) >= min_len else []
    return [s for s in np.split(data, np.flatnonzero(segment_starts)) if len(s) >= min_len]


def estimate_offsets(data, max_steer: float = 0.1, min_samples: int = 50):
    """Steering and yaw-rate offsets from driving with little steering.

    Fits the linear-tire yaw rate omega = k * v_x * (delta - delta_0) + omega_0 on the samples
    with |delta| < max_steer. omega_0 is only separable from delta_0 when the speed varies;
    with less than 1 m/s of speed spread it is taken as 0.

    Returns:
        delta_0 [rad], omega_0 [rad/s]; zeros when there is too little data for the fit.
    """
    near = np.abs(data[:, DELTA]) < max_steer
    if np.count_nonzero(near) < min_samples:
        return 0.0, 0.0
    v_x, omega, delta = data[near, V_X], data[near, OMEGA], data[near, DELTA]
    # shortcut: k is taken speed independent (no understeer gradient), fine for small offsets
    columns = [v_x * delta, v_x]
    speed_spread = np.percentile(v_x, 90) - np.percentile(v_x, 10) > 1.0
    if speed_spread:
        columns.append(np.ones_like(v_x))
    coeffs = np.linalg.lstsq(np.stack(columns, axis=1), omega, rcond=None)[0]
    k, c = coeffs[0], coeffs[1]
    if k <= 0:  # no usable steering response
        return 0.0, 0.0
    omega_0 = float(coeffs[2]) if speed_spread else 0.0
    return float(-c / k), omega_0
