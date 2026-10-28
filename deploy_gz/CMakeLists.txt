cmake_minimum_required(VERSION 3.8)
project(kairos_deploy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## 1. Dependencias de ROS 2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)

## 2. Buscar ONNX Runtime
find_package(onnxruntime QUIET)

if(NOT onnxruntime_FOUND)
  message(STATUS "onnxruntime config no encontrado. Buscando librerías manualmente...")
  
  find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h 
    PATHS /usr/include /usr/local/include /usr/include/onnxruntime /opt/onnxruntime/include
    REQUIRED
  )
  
  find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime 
    PATHS /usr/lib /usr/local/lib /opt/onnxruntime/lib
    REQUIRED
  )
endif()

## 3. Declarar los ejecutables (Nodos)
# --- Nodo 1: rl_inference_node ---
add_executable(rl_inference_node
  src/rl_inference_node.cpp
  observations/robot_observations.cpp
  include/kairos_deploy/deploy_config.cpp
)
target_include_directories(rl_inference_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/observations 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Nodo 2: tray_tilt_eval_node ---
add_executable(tray_tilt_eval_node
  src/tray_tilt_eval_node.cpp
  observations/robot_observations.cpp
  include/kairos_deploy/deploy_config.cpp
)
target_include_directories(tray_tilt_eval_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/observations 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

## 4. Enlazar dependencias de ROS 2
ament_target_dependencies(rl_inference_node
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_msgs
  ament_index_cpp
  tf2_geometry_msgs
)

ament_target_dependencies(tray_tilt_eval_node
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_msgs
  ament_index_cpp
  tf2_geometry_msgs
)

## 5. Enlazar dependencias de ONNX Runtime y yaml-cpp
if(onnxruntime_FOUND)
  target_link_libraries(rl_inference_node onnxruntime yaml-cpp)
  target_link_libraries(tray_tilt_eval_node onnxruntime yaml-cpp)
else()
  target_include_directories(rl_inference_node PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(rl_inference_node ${ONNXRUNTIME_LIB} yaml-cpp)
  
  target_include_directories(tray_tilt_eval_node PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(tray_tilt_eval_node ${ONNXRUNTIME_LIB} yaml-cpp)
endif()

## 6. Reglas de instalación para ROS 2
# Instalar los binarios de los nodos (Aquí SÍ se pueden poner varios TARGETS juntos)
install(TARGETS
  rl_inference_node
  tray_tilt_eval_node
  DESTINATION lib/${PROJECT_NAME}
)

# Instalar carpetas adicionales
install(DIRECTORY
  launch
  config
  models
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/spawn_cube_on_tray.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()