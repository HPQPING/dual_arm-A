"""安装实机反馈、位置控制节点和配置文件."""

from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = 'archer_hardware'

setup(
    name=PACKAGE_NAME,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'test.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml', 'README.md']),
        ('share/' + PACKAGE_NAME + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Archer 双臂官方 SDK 反馈与显式关节位置控制接入。',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'hardware_feedback = archer_hardware.driver_node:main',
            'hardware_position = archer_hardware.position_node:main',
            'hardware_circle = archer_hardware.circle_task:main',
            'hardware_grasp = archer_hardware.grasp_task:main',
            'hardware_operator = archer_hardware.operator_client:main',
        ],
    },
)
