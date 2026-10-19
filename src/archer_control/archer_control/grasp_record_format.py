"""抓取记录的公共字段声明；不导入仿真、ROS 或任务状态机."""

SIDES = ("left", "right")
JOINT_QUANTITIES = ("position_rad", "velocity_rad_s", "drive_input_rad")
OBJECT_COMPONENTS = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
EVENT_FIELDS = ("utc_time", "wall_elapsed_s", "stage")

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
