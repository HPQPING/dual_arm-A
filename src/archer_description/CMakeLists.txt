cmake_minimum_required(VERSION 3.10)

project(archer_description)

find_package(ament_cmake REQUIRED)

# 将模型、网格和显示配置安装到 ROS 2 包的资源目录。
install(
  DIRECTORY
    urdf
    meshes
    rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()