project(nbody-tpool VERSION 1.0 LANGUAGES CXX)

# define the project entry point source file
set(PROJECT_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}.cpp
)

# find the system threading library,
# prefer the pthreads implementation
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
# determine the threading library
find_package(Threads REQUIRED)

# build tpool.hpp as an interface library (header-only)
set(TPOOL_UTIL_DIR ${INCLUDE_DIR}/tpool)
add_library(tpool INTERFACE)
if(MSVC)
add_custom_target(tpool.headers SOURCES 
  ${TPOOL_UTIL_DIR}/tpool.hpp
)
endif()
target_include_directories(tpool INTERFACE ${TPOOL_UTIL_DIR})

# executable definition
add_executable(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE}
)

# link the executable with the threading library's generated imported target
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# graphics interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE graphics)

# sycl_utils interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE tpool)

# expose custom sandbox includes
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
