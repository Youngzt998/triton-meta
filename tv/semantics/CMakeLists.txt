# tile-smt — MLIR-free core semantic model. Links ONLY Z3 (no MLIR/Triton).
# This is the target the M0 migration grows; today it holds the neutral types.
# Z3 is already found by tv/CMakeLists.txt (z3::libz3 target / Z3_* vars).

add_library(tile-smt STATIC
  Types.cpp
  AbstractFp.cpp
  Memory.cpp
)

# Include root is tv/ so code uses #include "semantics/Types.h" (matches the
# existing include style), while the core lives under tv/semantics/.
target_include_directories(tile-smt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Z3 ONLY — deliberately no ${triton_libs}, no MLIRIR, no MLIRSupport.
if(TARGET z3::libz3)
  target_link_libraries(tile-smt PUBLIC z3::libz3)
else()
  target_include_directories(tile-smt PUBLIC ${Z3_INCLUDE_DIRS})
  target_link_libraries(tile-smt PUBLIC ${Z3_LIBRARIES})
endif()

option(TILE_SMT_BUILD_TESTS "Build tile-smt Z3-only unit tests" ON)
if(TILE_SMT_BUILD_TESTS)
  add_subdirectory(test)
endif()
