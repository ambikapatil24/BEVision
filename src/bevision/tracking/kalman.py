"""A constant-velocity 3D Kalman filter in pure numpy.

Replaces ``filterpy.kalman.KalmanFilter``. filterpy is unmaintained and pulls in a
dependency for what is ~40 lines of linear algebra, and owning the filter makes the
state layout and noise model explicit instead of hidden behind a general-purpose
class.

State is ``[x, y, z, vx, vy, vz]`` in the global frame, observed through position
only. The prediction step is the standard constant-velocity transition with
position updated by ``dt * velocity``; the update step uses the Joseph form for the
covariance, matching filterpy's implementation bit for bit.

Numerical equivalence with filterpy is asserted by ``tests/test_kalman.py`` when
filterpy is importable, so this replacement cannot silently change published
numbers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.config import KalmanConfig

FloatArray = NDArray[np.floating]

STATE_DIM = 6
MEASUREMENT_DIM = 3


class ConstantVelocityKalmanFilter:
    """A 3D constant-velocity Kalman filter over global-frame position.

    Args:
        initial_position: Starting ``(x, y, z)``. Velocity always starts at zero, so a new
            track's first prediction is a stationary one.
        dt: Seconds between consecutive samples.
        config: Noise parameters; the defaults match the reference ``filterpy`` configuration
            the committed results were produced with.
    """

    def __init__(
        self,
        initial_position: ArrayLike,
        dt: float,
        config: KalmanConfig | None = None,
    ) -> None:
        config = config or KalmanConfig()

        self._dt = float(dt)
        self._config = config

        self.x = np.zeros((STATE_DIM, 1), dtype=float)
        self.x[:MEASUREMENT_DIM, 0] = np.asarray(initial_position, dtype=float).reshape(3)

        self.F = np.eye(STATE_DIM, dtype=float)
        for i in range(MEASUREMENT_DIM):
            self.F[i, i + MEASUREMENT_DIM] = self._dt

        self.H = np.zeros((MEASUREMENT_DIM, STATE_DIM), dtype=float)
        self.H[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = np.eye(MEASUREMENT_DIM)

        # Initial state covariance: the identity scaled by config.initial_covariance (5.0 default).
        self.P = config.initial_covariance * np.eye(STATE_DIM, dtype=float)
        self.R = config.measurement_noise * np.eye(MEASUREMENT_DIM, dtype=float)
        self.Q = config.process_noise * np.eye(STATE_DIM, dtype=float)

        self._identity = np.eye(STATE_DIM, dtype=float)

    # -- accessors ----------------------------------------------------------
    @property
    def position(self) -> FloatArray:
        """Current position estimate, shape ``(3,)``."""
        return self.x[:MEASUREMENT_DIM, 0].copy()

    @property
    def velocity(self) -> FloatArray:
        """Current velocity estimate, shape ``(3,)``."""
        return self.x[MEASUREMENT_DIM:, 0].copy()

    @property
    def dt(self) -> float:
        return self._dt

    # -- filter steps -------------------------------------------------------
    def predict(self) -> None:
        """Advance the state by one sample period."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: ArrayLike) -> None:
        """Fold in a position measurement (shape ``(3,)``)."""
        z = np.asarray(measurement, dtype=float).reshape(MEASUREMENT_DIM, 1)

        innovation = z - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)

        self.x = self.x + gain @ innovation

        # Joseph form, as filterpy uses; more numerically stable than (I - KH) P.
        identity_minus_kh = self._identity - gain @ self.H
        self.P = identity_minus_kh @ self.P @ identity_minus_kh.T + gain @ self.R @ gain.T
