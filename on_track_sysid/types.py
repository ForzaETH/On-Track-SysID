from dataclasses import dataclass


@dataclass
class VehicleParams:
    """Single-track vehicle parameters. Field names match the race stack's CarConfig."""
    m: float
    """Mass [kg]"""
    Iz: float
    """Yaw moment of inertia [kg m^2]"""
    lf: float
    """Distance from the center of gravity to the front axle [m]"""
    lr: float
    """Distance from the center of gravity to the rear axle [m]"""
    h_cg: float
    """Height of the center of gravity [m]"""

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


@dataclass
class PacejkaParams:
    """Pacejka magic formula coefficients. Field names match the race stack's PacejkaTireConfig."""
    Bf: float
    Cf: float
    Df: float
    Ef: float
    Br: float
    Cr: float
    Dr: float
    Er: float
    friction_coeff: float = 1.0
    """Scales D of both axles; nominally 1.0"""

    @property
    def front(self) -> tuple:
        return (self.Bf, self.Cf, self.Df, self.Ef)

    @property
    def rear(self) -> tuple:
        return (self.Br, self.Cr, self.Dr, self.Er)

    def with_coeffs(self, front, rear) -> "PacejkaParams":
        """Returns a copy with new front and rear [B, C, D, E], keeping the friction coefficient."""
        return PacejkaParams(*map(float, front), *map(float, rear), friction_coeff=self.friction_coeff)
