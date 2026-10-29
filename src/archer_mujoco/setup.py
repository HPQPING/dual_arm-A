"""安装MuJoCo入口、YAML配置与README."""

from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "archer_mujoco"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        (
            "share/" + PACKAGE_NAME,
            ["package.xml", "README.md"],
        ),
        ("share/" + PACKAGE_NAME + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="user",
    maintainer_email="user@example.com",
    description=(
        "MuJoCo simulation and gripper coordinate visualization " "for Archer-Y6 robots."
    ),
    license="TODO",
    entry_points={
        "console_scripts": [
            "check_grasp = archer_mujoco.check_grasp:main",
            "dual_sim_node = archer_mujoco.dual_sim_node:main",
            "grasp_scene = archer_mujoco.grasp_scene:main",
            "gripper_markers = archer_mujoco.gripper_markers:main",
            "pose_check = archer_mujoco.pose_check:main",
            "sim_node = archer_mujoco.sim_node:main",
        ],
    },
)
