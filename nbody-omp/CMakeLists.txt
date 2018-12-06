project(nbody-omp VERSION 1.0 LANGUAGES CXX)

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
# find openmp
find_package(OpenMP)
# set c/c++ compiler flags for openmp
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# executable definition
add_executable(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE}
)

# link the executable with the threading library's generated imported target
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# graphics interface lib.
target_link_libraries(${PROJECT_NAME} PRIVATE graphics)

# expose custom sandbox includes
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
