cmake_minimum_required(VERSION 3.10)

project(archer_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    config
    rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # 工程不依赖第三方pytest插件；隔离用户额外插件与ROS自带pytest的版本冲突。
  set(_archer_pytest_autoload "$ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD}")
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(archer_bringup_regression test/test_circle_launch.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1)
  if(_archer_pytest_autoload STREQUAL "")
    unset(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD})
  else()
    set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "${_archer_pytest_autoload}")
  endif()
endif()

ament_package()