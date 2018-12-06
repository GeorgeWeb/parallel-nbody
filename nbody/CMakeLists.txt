project(nbody VERSION 1.0 LANGUAGES CXX)

# define the project entry point source file
set(PROJECT_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}.cpp
)

# executable definition
add_executable(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE}
)

# graphics interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE graphics)

# expose custom sandbox includes
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
