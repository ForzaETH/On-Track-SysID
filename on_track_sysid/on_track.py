from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from on_track_sysid.dynamics import lateral_dynamics
from on_track_sysid.pacejka import DEFAULT_BOUNDS, steady_state_forces
from on_track_sysid.preprocess import DELTA, OMEGA, V_X, V_Y, filter_data, mirror_data
from on_track_sysid.steady_state import fit_axles
from on_track_sysid.types import PacejkaParams, VehicleParams


@dataclass
class OnTrackConfig:
    num_of_iterations: int = 6
    num_of_epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.0
    dt: float = 0.02
    """Sample time of the data [s]"""
    vy_at_rear_axle: bool = True
    steer_delay_steps: int = 5
    sweep_steps: int = 500
    sweep_max_steer: float = 0.4
    sweep_vel_bounds: tuple = (2.75, 4.0)
    """The sweep runs at the mean data speed, clipped to these bounds [m/s]"""
    equal_tires: bool = False
    bounds: tuple = DEFAULT_BOUNDS
    seed: Optional[int] = None


def _make_nn():
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 8), nn.LeakyReLU(), nn.Linear(8, 8), nn.LeakyReLU(), nn.Linear(8, 2))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)
    return model


def _training_set(data, vehicle, tires, dt):
    """Inputs and one-step residuals of the nominal model, mirrored left-right."""
    v_x, v_y, omega, delta = (data[:-1, i] for i in (V_X, V_Y, OMEGA, DELTA))
    v_y_dot, omega_dot = lateral_dynamics(vehicle, tires, v_x, v_y, omega, delta)
    errors = np.stack((data[1:, V_Y] - (v_y + v_y_dot * dt),
                       data[1:, OMEGA] - (omega + omega_dot * dt)), axis=1)
    # the model is odd in (v_y, omega, delta), so the mirrored residual is the negated one
    return mirror_data(data[:-1]), np.concatenate((errors, -errors))


def _simulate_sweep(nn_model, vehicle, tires, v_x, cfg):
    """Constant-speed steering ramp with the nominal model plus the learned residual."""
    import torch

    n = cfg.sweep_steps
    v_x = np.full(n, v_x)
    v_y = np.zeros(n)
    omega = np.zeros(n)
    delta = np.linspace(0.0, cfg.sweep_max_steer, n)
    with torch.no_grad():
        for t in range(n - 1):
            v_y_dot, omega_dot = lateral_dynamics(vehicle, tires, v_x[t], v_y[t], omega[t], delta[t])
            residual = nn_model(torch.tensor([v_x[t], v_y[t], omega[t], delta[t]], dtype=torch.float32))
            v_y[t + 1] = v_y[t] + v_y_dot * cfg.dt + float(residual[0])
            omega[t + 1] = omega[t] + omega_dot * cfg.dt + float(residual[1])
    return v_x, v_y, omega, delta


def identify_on_track(data, vehicle: VehicleParams, init: PacejkaParams, cfg: OnTrackConfig = OnTrackConfig(),
                      callback: Optional[Callable[[int, PacejkaParams, tuple], None]] = None) -> PacejkaParams:
    """Identifies Pacejka coefficients from normal driving data.

    Each iteration trains a small network on the residual of the current nominal model,
    simulates a steady-state steering sweep with nominal model + network and refits the
    Pacejka coefficients on the sweep. The refit model is the next iteration's nominal model.

    Args:
        data: (N, 4) array of v_x, v_y, omega, delta sampled every cfg.dt seconds.
        init: initial nominal tire model; its friction_coeff is kept.
        callback: called after each iteration with (iteration, tires, (v_x, v_y, omega, delta) of the sweep).
    """
    import torch

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    data = filter_data(np.array(data, dtype=float), vehicle.lr, cfg.vy_at_rear_axle, cfg.steer_delay_steps)
    sweep_vel = np.clip(np.mean(data[:, V_X]), *cfg.sweep_vel_bounds)

    tires = init
    for i in range(1, cfg.num_of_iterations + 1):
        X, y = _training_set(data, vehicle, tires, cfg.dt)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        nn_model = _make_nn()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = torch.nn.MSELoss()
        nn_model.train()
        for _ in range(cfg.num_of_epochs):
            optimizer.zero_grad()
            loss = criterion(nn_model(X), y)
            loss.backward()
            optimizer.step()
        nn_model.eval()

        sweep = _simulate_sweep(nn_model, vehicle, tires, sweep_vel, cfg)
        alpha_f, alpha_r, F_zf, F_zr, F_yf, F_yr = steady_state_forces(vehicle, *sweep)
        # drop diverged samples of the sweep
        keep = (np.abs(sweep[OMEGA]) <= 5) & (np.abs(alpha_f) <= 0.5) & (np.abs(alpha_r) <= 0.2)
        tires = fit_axles(tires, alpha_f[keep], alpha_r[keep], F_zf, F_zr, F_yf[keep], F_yr[keep],
                          cfg.bounds, cfg.equal_tires)
        if callback is not None:
            callback(i, tires, sweep)
    return tires
