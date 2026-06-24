# Plain C++ unit tests for the tv translation validator.
#
# These use the tiny header-only harness in SimpleTest.h instead of GoogleTest,
# so they build with only the C++ standard library (no FetchContent, no
# network). Each test is a standalone executable; running it runs all of its
# TEST() cases and returns non-zero if any check fails. Tests are registered
# with CTest and grouped under the `tv-validator-tests` build target. When the
# Triton unit-test target exists, they are attached to it too.

enable_testing()

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

if(NOT TARGET tv-validator-tests)
  add_custom_target(tv-validator-tests)
endif()

# add_tv_test(NAME <name> SRCS <sources...> LIBS <libs...>)
function(add_tv_test)
  cmake_parse_arguments(ARG "" "NAME" "SRCS;LIBS" ${ARGN})

  add_executable(${ARG_NAME} ${ARG_SRCS})

  # current dir -> "SimpleTest.h"; tv/ root -> "semantics/Memory.h" etc.
  target_include_directories(${ARG_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
  )

  target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})

  # Z3 linking mirrors tv/CMakeLists.txt.
  if(TARGET z3::libz3)
    target_link_libraries(${ARG_NAME} PRIVATE z3::libz3)
  else()
    target_include_directories(${ARG_NAME} PRIVATE ${Z3_INCLUDE_DIRS})
    target_link_libraries(${ARG_NAME} PRIVATE ${Z3_LIBRARIES})
  endif()

  add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
  add_dependencies(tv-validator-tests ${ARG_NAME})
  if(TARGET TritonUnitTests)
    add_dependencies(TritonUnitTests ${ARG_NAME})
  endif()
endfunction()

add_tv_test(
  NAME TestTritonTVMemory
  SRCS MemoryModelTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
  LIBS
    MLIRIR
    MLIRSupport
)

add_tv_test(
  NAME TestTritonTVEnv
  SRCS EnvTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Env.cpp
  LIBS
    ${triton_libs}
    MLIRIR
    MLIRSupport
)

add_tv_test(
  NAME TestTritonTVState
  SRCS StateTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/AbstractFp.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Memory.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/Env.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/State.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/mlir/ArithOps.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/mlir/TritonOps.cpp
  LIBS
    ${triton_libs}
    MLIRIR
    MLIRSupport
)

add_tv_test(
  NAME TestTritonTVAbstractFp
  SRCS AbstractFpTest.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../../semantics/AbstractFp.cpp
  LIBS
    MLIRIR
    MLIRSupport
)
