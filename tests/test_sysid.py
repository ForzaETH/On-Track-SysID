import numpy as np
import pytest

from on_track_sysid import (OnTrackConfig, PacejkaParams, VehicleParams, fit_pacejka, generate_lut,
                            identify_on_track, identify_steady_state, lateral_dynamics, pacejka_formula)
from on_track_sysid.pacejka import axle_loads
from on_track_sysid.preprocess import estimate_offsets, split_segments

# NUC6 on the dubi floor
VEHICLE = VehicleParams(m=3.39, Iz=0.05797, lf=0.168, lr=0.167, h_cg=0.014)
TIRES = PacejkaParams(Bf=7.1084, Cf=0.683, Df=2.0, Ef=0.445, Br=4.9599, Cr=1.5278, Dr=1.1269, Er=1.1)
GUESS = PacejkaParams(Bf=5.0, Cf=1.0, Df=1.5, Ef=0.5, Br=5.0, Cr=1.0, Dr=1.5, Er=0.5)


def force_error(tires, ref, alpha):
    """Largest front/rear force difference on the given slip-angle range [N]."""
    F_zf, F_zr = axle_loads(VEHICLE)
    return max(np.max(np.abs(pacejka_formula(tires.front, alpha, F_zf) - pacejka_formula(ref.front, alpha, F_zf))),
               np.max(np.abs(pacejka_formula(tires.rear, alpha, F_zr) - pacejka_formula(ref.rear, alpha, F_zr))))


def test_friction_coeff_scales_force():
    alpha = np.linspace(-0.4, 0.4, 41)
    full = pacejka_formula(TIRES.front, alpha, 10.0)
    np.testing.assert_allclose(pacejka_formula(TIRES.front, alpha, 10.0, mu=0.5), 0.5 * full)


def test_fit_pacejka_recovers_coefficients():
    alpha = np.linspace(-0.5, 0.5, 201)
    true = (7.0, 1.5, 1.2, 0.4)  # well inside the bounds; the NUC6 fit sits on them
    coeffs = fit_pacejka(alpha, 15.0, pacejka_formula(true, alpha, 15.0), GUESS.rear, fit_mask=(True,) * 4)
    np.testing.assert_allclose(coeffs, true, rtol=1e-3)


def test_fit_pacejka_keeps_unfitted_coefficients():
    alpha = np.linspace(-0.5, 0.5, 201)
    coeffs = fit_pacejka(alpha, 15.0, pacejka_formula((7.0, 1.5, 1.2, 0.4), alpha, 15.0), GUESS.rear)
    assert coeffs[3] == pytest.approx(GUESS.Er)  # E is not fitted by default


def test_estimate_offsets():
    rng = np.random.default_rng(0)
    v_x = rng.uniform(1.5, 5.0, 2000)
    delta = rng.uniform(-0.08, 0.08, 2000)
    omega = 3.0 * v_x * (delta - 0.02) + 0.05 + rng.normal(0.0, 0.01, 2000)
    data = np.stack((v_x, np.zeros_like(v_x), omega, delta), axis=1)
    np.testing.assert_allclose(estimate_offsets(data), (0.02, 0.05), atol=2e-3)


def test_split_segments():
    data = np.arange(10.0)[:, None]
    starts = np.zeros(10, dtype=bool)
    starts[[0, 3, 4]] = True
    segments = split_segments(data, starts, min_len=2)
    assert [s[:, 0].tolist() for s in segments] == [[0, 1, 2], [4, 5, 6, 7, 8, 9]]


def steering_ramp(v_x=3.0, n=3000, dt=0.02, max_steer=0.3):
    """Slow steering ramp at constant speed, integrated with the true model."""
    delta = np.linspace(0.0, max_steer, n)
    v_y = np.zeros(n)
    omega = np.zeros(n)
    for t in range(n - 1):
        v_y_dot, omega_dot = lateral_dynamics(VEHICLE, TIRES, v_x, v_y[t], omega[t], delta[t])
        v_y[t + 1] = v_y[t] + v_y_dot * dt
        omega[t + 1] = omega[t] + omega_dot * dt
    return np.stack((np.full(n, v_x), v_y, omega, delta), axis=1)


def test_identify_steady_state():
    tires = identify_steady_state(steering_ramp(), VEHICLE, GUESS, vy_at_rear_axle=False)
    assert force_error(tires, TIRES, np.linspace(-0.1, 0.1, 21)) < 0.5


def test_lut_monotonic_until_max():
    table = generate_lut(VEHICLE, TIRES, steers=np.linspace(0, 0.4, 21), vels=[1.0, 3.0, 5.0])
    assert table.shape == (22, 4)
    for j in range(1, 4):
        a_lat = table[1:, j]
        a_lat = a_lat[~np.isnan(a_lat)]
        assert a_lat.size > 1
        assert np.all(np.diff(a_lat) >= 0)
    # low speed steady state matches the kinematic yaw rate v * tan(delta) / l
    np.testing.assert_allclose(table[2, 1], 1.0 ** 2 * np.tan(0.02) / VEHICLE.wheelbase, rtol=0.05)


def test_identify_on_track_improves_model():
    pytest.importorskip("torch")
    n, dt = 2500, 0.02
    t = np.arange(n) * dt
    delta = 0.12 * np.sin(2 * np.pi * 0.25 * t)
    data = np.zeros((n, 4))
    data[:, 0] = 3.0
    data[:, 3] = delta
    for k in range(n - 1):
        v_y_dot, omega_dot = lateral_dynamics(VEHICLE, TIRES, 3.0, data[k, 1], data[k, 2], delta[k])
        data[k + 1, 1] = data[k, 1] + v_y_dot * dt
        data[k + 1, 2] = data[k, 2] + omega_dot * dt

    cfg = OnTrackConfig(num_of_iterations=3, vy_at_rear_axle=False, steer_delay_steps=0, seed=0)
    tires = identify_on_track(data, VEHICLE, GUESS, cfg)
    alpha = np.linspace(-0.08, 0.08, 17)
    assert force_error(tires, TIRES, alpha) < force_error(GUESS, TIRES, alpha)


def test_lut_rejects_bad_grids():
    with pytest.raises(ValueError):
        generate_lut(VEHICLE, TIRES, steers=[-0.1, 0.0, 0.1], vels=[3.0])
    with pytest.raises(ValueError):
        generate_lut(VEHICLE, TIRES, steers=[0.2, 0.1], vels=[3.0])
    with pytest.raises(ValueError):
        generate_lut(VEHICLE, TIRES, steers=[0.0, 0.1], vels=[3.0], duration=0.1, dt=0.01)
