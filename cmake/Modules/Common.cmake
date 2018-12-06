# For MSVC, set the compiler flags and properties
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# For Clang, increase the max number of constexpr steps. A lot.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=2147483647")
endif()

# Sets console/terminal compilation colored output for the ninja generator
# source: https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COMPILER_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)
if (${FORCE_COMPILER_COLORED_OUTPUT})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif ()
endif ()
