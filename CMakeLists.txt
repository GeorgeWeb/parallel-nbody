cmake_minimum_required(VERSION 3.10.2)
# change to multi-parallel-nbody
project(gpu-nbody VERSION 0.1 LANGUAGES C CXX)

# store single header files and own libraries here
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# graphics header library file directories
set(GRAPHICS_ROOT_DIR "${INCLUDE_DIR}/graphics")
set(GRAPHICS_VENDOR_DIR "${GRAPHICS_ROOT_DIR}/vendor")

# C++ [minimum] standard version (required!)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# append custom path to the default cmake modules path variable
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
# compilation setup and utilities
include(Common)

add_library(glad STATIC ${GRAPHICS_VENDOR_DIR}/glad/src/glad.c)
# generate GLAD static library
target_include_directories(glad PRIVATE ${GRAPHICS_VENDOR_DIR}/glad/include)

# generate GLFW static library (from source)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${GRAPHICS_VENDOR_DIR}/glfw)

# build graphics project as an interface library (header-only)
add_library(graphics INTERFACE)
if(MSVC)
add_custom_target(graphics.headers SOURCES 
  ${GRAPHICS_ROOT_DIR}/graphics.hpp
  ${GRAPHICS_ROOT_DIR}/app/app.hpp
  ${GRAPHICS_ROOT_DIR}/window/window.hpp
  ${GRAPHICS_ROOT_DIR}/camera/camera.hpp
  ${GRAPHICS_ROOT_DIR}/shader/shader.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/renderer.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/mesh.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/vertex_array.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/vertex_buffer.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/vertex_buffer_layout.hpp
  ${GRAPHICS_ROOT_DIR}/renderer/index_buffer.hpp
  ${GRAPHICS_ROOT_DIR}/util/glutil.hpp
)
endif()
# expose custom graphics includes
target_include_directories(graphics INTERFACE ${GRAPHICS_ROOT_DIR})
# link GLFW (glfw3)
target_link_libraries(graphics INTERFACE glfw)
target_include_directories(graphics INTERFACE ${GRAPHICS_VENDOR_DIR}/glfw/include)
# link GLAD (glad2)
target_link_libraries(graphics INTERFACE glad)
target_include_directories(graphics INTERFACE ${GRAPHICS_VENDOR_DIR}/glad/include)
# link GLM (glsl-like math library for opengl)
target_include_directories(graphics INTERFACE ${GRAPHICS_VENDOR_DIR}/glm/include)
# link DEBUGBREAK (used to programmatically break into debugger)
target_include_directories(graphics INTERFACE ${GRAPHICS_VENDOR_DIR}/debugbreak/include)

# generate the sandbox apps
add_subdirectory(nbody)        # nbody simulation app
add_subdirectory(nbody-omp)    # OpenMP nbody simulation app
add_subdirectory(nbody-tpool)  # Thread-Pool nbody simulation app
add_subdirectory(nbody-sycl)   # OpenCL SYCL nbody simulation app
