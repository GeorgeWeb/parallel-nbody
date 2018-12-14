project(nbody-sycl VERSION 1.0 LANGUAGES CXX)

# find installed ComputeCpp lib.
find_package(ComputeCpp REQUIRED)

# define the project entry point source file
set(PROJECT_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}.cpp
)

# build sycl_utils as an interface library (header-only)
set(SYCL_UTILS_DIR ${INCLUDE_DIR}/sycl_utils)
add_library(sycl_utils INTERFACE)
if(MSVC)
add_custom_target(sycl_utils.headers SOURCES 
  ${SYCL_UTILS_DIR}/sycl_utils.hpp
  ${SYCL_UTILS_DIR}/device/device_util.hpp
  ${SYCL_UTILS_DIR}/memory/memory_util.hpp
  ${SYCL_UTILS_DIR}/profiling/profiling_util.hpp
)
endif()
target_include_directories(sycl_utils INTERFACE ${SYCL_UTILS_DIR})

# executable definition
add_executable(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE}
)

# expose custom sandbox includes
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# graphics interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE graphics)

# sycl_utils interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE sycl_utils)

# link ComputeCpp (SYCL runtime lib.)
add_sycl_to_target(
  TARGET ${PROJECT_NAME}
  SOURCES ${PROJECT_SOURCE}
)
