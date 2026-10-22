cmake_minimum_required(VERSION 3.8)
project(kv_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SmokeDetection.msg"
  DEPENDENCIES std_msgs
)

# Install handwritten modules beside this package's generated message bindings.
install(DIRECTORY ${PROJECT_NAME}/
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(PROGRAMS scripts/smoke_detector.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(smoke_detection test/test_smoke_detection.py TIMEOUT 30)
  ament_add_pytest_test(smoke_background test/test_smoke_background.py TIMEOUT 30)
  ament_add_pytest_test(smoke_detector test/test_smoke_detector.py TIMEOUT 45)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
