"""Tests for the numpy Kalman filter.

The important test here is ``test_matches_filterpy_step_by_step``: it runs the
replacement and filterpy side by side on identical inputs and asserts the state and
covariance agree at every step. Without it, swapping the filter could silently change
published metrics, which is exactly the kind of regression the reproduction check in
the README exists to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.config import KalmanConfig
from bevision.tracking.kalman import ConstantVelocityKalmanFilter

DT = 0.5


def test_initial_state_is_zero_velocity_at_the_spawn_position() -> None:
    kf = ConstantVelocityKalmanFilter([1.0, -2.0, 3.0], dt=DT)
    assert np.allclose(kf.position, [1.0, -2.0, 3.0])
    assert np.allclose(kf.velocity, [0.0, 0.0, 0.0])


def test_noise_matrices_match_the_original_filterpy_setup() -> None:
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    assert np.allclose(kf.P, 5.0 * np.eye(6))  # filterpy P = I, then *= 5.0
    assert np.allclose(kf.R, np.eye(3))
    assert np.allclose(kf.Q, 0.01 * np.eye(6))


def test_transition_matrix_uses_dt_for_the_velocity_coupling() -> None:
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    expected = np.eye(6)
    for i in range(3):
        expected[i, i + 3] = DT
    assert np.allclose(kf.F, expected)


def test_observation_matrix_selects_position_only() -> None:
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    expected = np.zeros((3, 6))
    expected[:3, :3] = np.eye(3)
    assert np.allclose(kf.H, expected)


def test_custom_noise_config_is_respected() -> None:
    config = KalmanConfig(initial_covariance=2.0, measurement_noise=3.0, process_noise=0.5)
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT, config=config)
    assert np.allclose(kf.P, 2.0 * np.eye(6))
    assert np.allclose(kf.R, 3.0 * np.eye(3))
    assert np.allclose(kf.Q, 0.5 * np.eye(6))


def test_constant_velocity_is_learned_from_measurements() -> None:
    """A steadily moving target should end up with a velocity estimate near truth."""
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    speed = 4.0
    for step in range(1, 21):
        kf.predict()
        kf.update([speed * DT * step, 0.0, 0.0])

    assert kf.velocity[0] == pytest.approx(speed, rel=0.05)
    assert kf.velocity[1] == pytest.approx(0.0, abs=1e-6)


def test_position_tracks_a_measurement_immediately() -> None:
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    kf.predict()
    kf.update([10.0, 0.0, 0.0])
    # the gain is well below 1, so the estimate moves toward but not onto the measurement
    assert 0.0 < kf.position[0] < 10.0


def test_predict_does_not_move_position_without_velocity() -> None:
    kf = ConstantVelocityKalmanFilter([5.0, 5.0, 5.0], dt=DT)
    kf.predict()
    assert np.allclose(kf.position, [5.0, 5.0, 5.0])


def test_update_rejects_wrong_measurement_shape() -> None:
    kf = ConstantVelocityKalmanFilter([0.0, 0.0, 0.0], dt=DT)
    with pytest.raises(ValueError):
        kf.update([1.0, 2.0])


def test_matches_filterpy_step_by_step() -> None:
    """The replacement must be numerically identical to filterpy's implementation."""
    filterpy_kalman = pytest.importorskip("filterpy.kalman")

    rng = np.random.default_rng(20260929)
    spawn = np.array([12.5, -3.25, 0.75])

    # -- filterpy, configured exactly as the original notebook did --
    reference = filterpy_kalman.KalmanFilter(dim_x=6, dim_z=3)
    reference.x = np.array([[spawn[0]], [spawn[1]], [spawn[2]], [0.0], [0.0], [0.0]])
    reference.F = np.eye(6)
    for i in range(3):
        reference.F[i, i + 3] = DT
    reference.H = np.zeros((3, 6))
    reference.H[0, 0] = reference.H[1, 1] = reference.H[2, 2] = 1.0
    reference.P = reference.P * 5.0
    reference.R = np.eye(3) * 1.0
    reference.Q = np.eye(6) * 0.01

    replacement = ConstantVelocityKalmanFilter(spawn, dt=DT)

    assert np.allclose(replacement.P, reference.P)

    for step in range(40):
        reference.predict()
        replacement.predict()
        assert np.allclose(replacement.x, reference.x, atol=1e-9), (
            f"state diverged during predict at step {step}"
        )
        assert np.allclose(replacement.P, reference.P, atol=1e-9), (
            f"covariance diverged during predict at step {step}"
        )

        # feed both the same measurement, with occasional misses (predict-only frames)
        if step % 5 != 3:
            measurement = spawn + np.array([step * 0.5, 0.0, 0.0]) + rng.normal(0, 0.3, 3)
            reference.update(measurement)
            replacement.update(measurement)
            assert np.allclose(replacement.x, reference.x, atol=1e-9), (
                f"state diverged during update at step {step}"
            )
            assert np.allclose(replacement.P, reference.P, atol=1e-9), (
                f"covariance diverged during update at step {step}"
            )
