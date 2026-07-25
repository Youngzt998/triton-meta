# Z3-only unit tests for the tile-smt core. Each test links ONLY `tile-smt`
# (which brings the tv/ include root + Z3); NO MLIR/Triton. Grouped under the
# `tile-smt-tests` build target and registered with CTest.

if(NOT TARGET tile-smt-tests)
  add_custom_target(tile-smt-tests)
endif()

function(add_tile_smt_test name)
  cmake_parse_arguments(ARG "" "" "SRCS" ${ARGN})
  add_executable(${name} ${ARG_SRCS})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # SimpleTest.h
  target_link_libraries(${name} PRIVATE tile-smt)                          # core + Z3 (no MLIR)
  add_test(NAME ${name} COMMAND ${name})
  add_dependencies(tile-smt-tests ${name})
endfunction()

add_tile_smt_test(TileSmtTypesTest SRCS TypesTest.cpp)
add_tile_smt_test(TileSmtAbstractFpTest SRCS AbstractFpTest.cpp)
add_tile_smt_test(TileSmtMemoryTest SRCS MemoryTest.cpp)
add_tile_smt_test(TileSmtContextTest SRCS ContextTest.cpp)
add_tile_smt_test(TileSmtEquivalenceTest SRCS EquivalenceTest.cpp)
