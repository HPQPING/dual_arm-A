from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_NAME = 'archer_mujoco'


def resource_files(directory):
    """保留资源目录结构，安装其中的非隐藏文件。"""
    result = []
    root = Path(directory)

    if not root.is_dir():
        return result

    for parent in sorted(
        {path.parent for path in root.rglob('*') if path.is_file()}
    ):
        files = sorted(
            str(path)
            for path in parent.iterdir()
            if path.is_file() and not path.name.startswith('.')
        )

        if files:
            destination = (
                Path('share') / PACKAGE_NAME / parent
            ).as_posix()
            result.append((destination, files))

    return result


setup(
    name=PACKAGE_NAME,
    version='0.1.0',

    packages=find_packages(exclude=['test', 'test.*']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + PACKAGE_NAME],
        ),
        (
            'share/' + PACKAGE_NAME,
            ['package.xml'],
        ),
        *resource_files('config'),
        *resource_files('models'),
    ],

    install_requires=['setuptools'],
    zip_safe=False,

    maintainer='user',
    maintainer_email='user@example.com',

    description=(
        'MuJoCo simulation and gripper coordinate visualization '
        'for Archer-Y6 robots.'
    ),

    license='TODO',

    entry_points={
        'console_scripts': [
            'check_grasp = archer_mujoco.check_grasp:main',
            'dual_model_loader = archer_mujoco.dual_model_loader:main',
            'dual_sim_node = archer_mujoco.dual_sim_node:main',
            'grasp_scene = archer_mujoco.grasp_scene:main',
            'gripper_markers = archer_mujoco.gripper_markers:main',
            'model_loader = archer_mujoco.model_loader:main',
            'pose_check = archer_mujoco.pose_check:main',
            'sim_node = archer_mujoco.sim_node:main',
        ],
    },
)