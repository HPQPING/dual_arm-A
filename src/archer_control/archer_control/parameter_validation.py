"""公共数值参数检查，不导入 ROS 节点或启动任务."""

from array import array
import math


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def numbers(value: object, name: str) -> list[float]:
    if not isinstance(value, (list, tuple, array)):
        raise ValueError(f"{name} must be an array of numbers")
    return [number(item, name) for item in value]
