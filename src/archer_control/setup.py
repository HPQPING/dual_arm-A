"""安装轨迹规划、仿真任务入口与配置资源，不启动任务."""

from glob import glob
from setuptools import find_packages, setup


PACKAGE_NAME = "archer_control"


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
        (
            "share/" + PACKAGE_NAME + "/config",
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Trajectory planning and task execution for Archer-Y6 arms.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "single_arm_task = archer_control.single_arm_node:main",
            "dual_grasp_scene = archer_control.dual_grasp_scene:main",
        ],
    },
)
