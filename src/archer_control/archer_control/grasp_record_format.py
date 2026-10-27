"""抓取记录的公共字段声明；不导入仿真、ROS 或任务状态机."""

SIDES = ("left", "right")
JOINT_QUANTITIES = ("position_rad", "velocity_rad_s", "drive_input_rad")
OBJECT_COMPONENTS = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
EVENT_FIELDS = ("utc_time", "wall_elapsed_s", "stage")

# 用户确认的简化演示规则；毫米量不直接作为 SDK 电机位置发送。
SIMPLE_GRASP_RULE_VERSION = 4
INSERTION_DISTANCE_M = 0.01
GRIPPER_CLOSED_COMMAND = 5.65
GRIPPER_OPEN_COMMAND = 5.5
GRIPPER_ACTION_S = 1.0

SIMPLE_GRASP_STAGES = (
    "READY", "PLANNING", "SETTLE", "APPROACH", "APPROACH_SETTLE",
    "ALIGN_JOINT6", "ALIGN_SETTLE", "OPEN_FOR_GRASP",
    "INSERT", "INSERT_SETTLE", "LIFT", "MOVE", "RETURN", "LOWER",
    "PLACE_SETTLE", "RETREAT", "RETREAT_SETTLE", "CLOSE_BEFORE_HOME",
    "RETURN_HOME", "HOME_SETTLE", "COMPLETED",
)

# 保持已有 CSV 的顺序和名称，旧记录仍可直接验收与回放。
STATE_FIELDS = (
    "wall_elapsed_s",
    "simulation_time_s",
    "stage",
    "gripper_command",
    "virtual_attached",
    *(
        f"{side}_joint_{index}_{quantity}"
        for quantity in JOINT_QUANTITIES
        for side in SIDES
        for index in range(1, 7)
    ),
    *("object_" + component for component in OBJECT_COMPONENTS),
    *(f"{side}_tcp_{axis}_m" for side in SIDES for axis in ("x", "y", "z")),
)
