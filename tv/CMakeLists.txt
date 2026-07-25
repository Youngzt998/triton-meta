set(Z3_ROOT "" CACHE PATH "Path to custom Z3 installation (searched first)")

# Try custom location first, then fall back to system
if(Z3_ROOT)
  find_package(Z3 4.8 CONFIG QUIET PATHS "${Z3_ROOT}" NO_DEFAULT_PATH)
  if(NOT Z3_FOUND)
    find_path(Z3_INCLUDE_DIRS NAMES z3++.h PATHS "${Z3_ROOT}/include" NO_DEFAULT_PATH)
    find_library(Z3_LIBRARIES NAMES z3 PATHS "${Z3_ROOT}/lib" NO_DEFAULT_PATH)
  endif()
endif()

# Fall back to system
if(NOT Z3_FOUND AND NOT (Z3_INCLUDE_DIRS AND Z3_LIBRARIES))
  find_package(Z3 4.8 CONFIG QUIET)
  if(NOT Z3_FOUND)
    find_path(Z3_INCLUDE_DIRS NAMES z3++.h)
    find_library(Z3_LIBRARIES NAMES z3)
  endif()
endif()

if(NOT Z3_FOUND AND NOT (Z3_INCLUDE_DIRS AND Z3_LIBRARIES))
  message(FATAL_ERROR "Z3 not found. Install libz3-dev or set -DZ3_ROOT=/path/to/z3.")
endif()

# ---------------------------------------------------------------------------
# tile-smt (M0) — MLIR-free core semantic model, links ONLY z3.
# ---------------------------------------------------------------------------
add_subdirectory(semantics)

# ---------------------------------------------------------------------------
# Builder layer (per-language modeling onto the core):
#   tile-smt-builder-mlir    — shared MLIR tools + arith/math/scf handlers
#   tile-smt-builder-triton  — Triton tt.* handlers (links builder-mlir)
# Both compile WITHOUT -fno-exceptions/-fno-rtti (the default; we do NOT apply
# TRITON_DISABLE_EH_RTTI_FLAGS to them) so std::throw (Env::lookup) and z3++
# exceptions work — matching the old TritonTVSemantics.
# ---------------------------------------------------------------------------
add_subdirectory(builder/mlir)
add_subdirectory(builder/triton)

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# ---------------------------------------------------------------------------
# triton-tv binary
# ---------------------------------------------------------------------------
add_executable(triton-tv bin/triton-tv.cpp)

target_compile_options(triton-tv PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-tv PRIVATE
  tile-smt-builder-triton   # core + builder-mlir + Triton tt.* handlers
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  MLIRParser
  MLIRSupport
)

if(TARGET z3::libz3)
  target_link_libraries(triton-tv PRIVATE z3::libz3)
else()
  target_include_directories(triton-tv PRIVATE ${Z3_INCLUDE_DIRS})
  target_link_libraries(triton-tv PRIVATE ${Z3_LIBRARIES})
endif()

mlir_check_all_link_libraries(triton-tv)

# Validator unit tests use the header-only harness in
# tv/test/validator/SimpleTest.h (no GoogleTest, no network). Allow turning
# them off to build just the triton-tv binary.
option(TRITON_TV_BUILD_TESTS "Build tv validator unit tests" ON)
if(TRITON_TV_BUILD_TESTS)
  add_subdirectory(test/validator)
endif()
