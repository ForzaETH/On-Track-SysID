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
