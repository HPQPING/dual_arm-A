"""Joint trajectory planning. No hardware access; ROS imports are lazy.

Units: radians, radians/second, seconds. A quintic curve is used for each
segment, with zero target velocity and acceleration at each waypoint.
This provides joint-space paths, not Cartesian straight lines or circles.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from trajectory_msgs.msg import JointTrajectory

FloatArray: TypeAlias = NDArray[np.float64]


# Defaults match the archer_arm.xacro supplied in this project.
# They are model limits, not newly verified hardware specifications.
DEFAULT_JOINT_LIMITS = (
    (-2.79, 2.79), (-1.57, 2.09), (0.0, 3.14),
    (-1.57, 1.57), (-1.55, 1.55), (-2.96, 2.96),
)
DEFAULT_SPEED_LIMITS = (0.1, 0.5, 0.5, 0.5, 0.5, 0.5)
MAX_TOTAL_DURATION = 3600.0


def joint_names(side: str) -> tuple[str, ...]:
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    return tuple(f'{side}_joint_{index}' for index in range(1, 7))


def _number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def joint_vector(values: ArrayLike, name: str = 'joints') -> FloatArray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (6,):
        raise ValueError(f'{name} must contain exactly six numbers')
    return np.asarray([_number(value, name) for value in raw], dtype=np.float64)


def _speed_vector(values: ArrayLike) -> FloatArray:
    result = joint_vector(values, 'speed_limits')
    if np.any(result <= 0):
        raise ValueError('Every speed limit must be positive')
    return result


def _limit_matrix(values: ArrayLike) -> FloatArray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (6, 2):
        raise ValueError('joint_limits must have shape (6, 2)')
    result = np.asarray([
        [_number(value, 'joint_limits') for value in row] for row in raw
    ], dtype=np.float64)
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError('Each lower limit must be less than its upper limit')
    return result


def validate_joints(
    values: ArrayLike,
    limits: ArrayLike = DEFAULT_JOINT_LIMITS,
) -> FloatArray:
    q = joint_vector(values)
    bounds = _limit_matrix(limits)
    bad = np.flatnonzero((q < bounds[:, 0]) | (q > bounds[:, 1]))
    if len(bad):
        index = int(bad[0])
        raise ValueError(
            f'J{index + 1}={q[index]:.6f} is outside '
            f'[{bounds[index, 0]:.6f}, {bounds[index, 1]:.6f}]'
        )
    return q


def minimum_segment_duration(
    start: ArrayLike,
    target: ArrayLike,
    speed_limits: ArrayLike = DEFAULT_SPEED_LIMITS,
) -> float:
    """Minimum duration for a quintic segment's target velocity limits.

    The normalized quintic's maximum slope is 1.875. This check does not
    impose acceleration/jerk limits or guarantee hardware tracking.
    """
    q0 = joint_vector(start, 'start')
    q1 = joint_vector(target, 'target')
    speeds = _speed_vector(speed_limits)
    return float(np.max(1.875 * np.abs(q1 - q0) / speeds))


@dataclass(frozen=True)
class JointPlan:
    joint_names: tuple[str, ...]
    times: FloatArray
    positions: FloatArray

    def __post_init__(self):
        names = tuple(self.joint_names)
        times = np.asarray(self.times, dtype=np.float64).copy()
        positions = np.asarray(self.positions, dtype=np.float64).copy()
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError('A plan must contain six unique joint names')
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Joint names must be non-empty strings')
        if times.ndim != 1 or not 2 <= len(times) <= 10000:
            raise ValueError('A plan must contain 2 to 10000 time points')
        if positions.shape != (len(times), 6):
            raise ValueError('positions must have shape (number_of_times, 6)')
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError('Plan times and positions must be finite')
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Times must start at zero and strictly increase')
        if times[-1] > MAX_TOTAL_DURATION:
            raise ValueError('Total duration must not exceed 3600 seconds')
        times.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, 'joint_names', names)
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'positions', positions)

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def plan_joint_path(
    side: str,
    start: ArrayLike,
    waypoints: Iterable[ArrayLike],
    speed_limits: ArrayLike = DEFAULT_SPEED_LIMITS,
    joint_limits: ArrayLike = DEFAULT_JOINT_LIMITS,
    minimum_duration: float = 4.0,
    time_margin: float = 1.1,
    segment_durations: Iterable[float] | None = None,
) -> JointPlan:
    """Plan from the current joint feedback through one or more waypoints.

    With no explicit segment_durations, choose the larger of minimum_duration
    and the speed-limited duration with time_margin. Explicit durations are
    checked and rejected if too short; they are never silently shortened.
    """
    names = joint_names(side)
    speeds = _speed_vector(speed_limits)
    bounds = _limit_matrix(joint_limits)
    minimum = _number(minimum_duration, 'minimum_duration')
    margin = _number(time_margin, 'time_margin')
    if minimum <= 0 or margin < 1.0:
        raise ValueError('minimum_duration must be positive; time_margin must be >= 1')
    points = list(waypoints)
    if not 1 <= len(points) <= 9999:
        raise ValueError('Provide 1 to 9999 waypoints')
    durations = None if segment_durations is None else list(segment_durations)
    if durations is not None and len(durations) != len(points):
        raise ValueError('Provide one duration for each waypoint')

    positions = [validate_joints(start, bounds)]
    times = [0.0]
    for index, waypoint in enumerate(points):
        target = validate_joints(waypoint, bounds)
        required = minimum_segment_duration(positions[-1], target, speeds)
        if durations is None:
            duration = max(minimum, required * margin)
        else:
            duration = _number(durations[index], 'segment_duration')
            if duration <= 0 or duration < required:
                raise ValueError(
                    f'Segment {index + 1} needs at least {required:.6f} seconds; '
                    f'given {duration:.6f}'
                )
        # At least one nanosecond between ROS message time points.
        if duration < 1e-9:
            raise ValueError('Each segment must last at least one nanosecond')
        times.append(times[-1] + duration)
        positions.append(target)
    return JointPlan(
        names,
        np.asarray(times, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
    )


def sample_joint_plan(plan: JointPlan, elapsed: float) -> tuple[FloatArray, bool]:
    """Sample a joint plan; values beyond its end hold the final target."""
    t = max(0.0, _number(elapsed, 'elapsed'))
    if t >= plan.duration:
        return plan.positions[-1].copy(), True
    index = int(np.searchsorted(plan.times, t, side='right')) - 1
    u = (t - plan.times[index]) / (plan.times[index + 1] - plan.times[index])
    s = 6 * u**5 - 15 * u**4 + 10 * u**3
    position = plan.positions[index] + s * (
        plan.positions[index + 1] - plan.positions[index]
    )
    return position, False


def stretch_plan(plan: JointPlan, total_duration: float) -> JointPlan:
    """Lengthen a plan, e.g. to align two arms' total trajectory durations.

    This does not synchronize start times. The coordination node must do that.
    """
    duration = _number(total_duration, 'total_duration')
    if duration < plan.duration:
        raise ValueError('stretch_plan cannot shorten a trajectory')
    return JointPlan(
        plan.joint_names,
        plan.times * (duration / plan.duration),
        plan.positions,
    )


def to_ros_trajectory(plan: JointPlan) -> JointTrajectory:
    """Build the positions-only message accepted by the current sim_node.

    A zero header timestamp means start when received. This is not a
    hardware-timed dual-arm synchronization interface or an action server.
    """
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = list(plan.joint_names)
    points: list[JointTrajectoryPoint] = []
    previous_ns = -1
    for time, position in zip(plan.times, plan.positions):
        nanoseconds = int(round(float(time) * 1_000_000_000))
        if nanoseconds <= previous_ns:
            raise ValueError('Time points are too close after nanosecond rounding')
        previous_ns = nanoseconds
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in position]
        point.time_from_start.sec = nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = nanoseconds % 1_000_000_000
        points.append(point)
    message.points = points
    return message