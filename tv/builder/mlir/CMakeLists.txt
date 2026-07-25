# tile-smt-builder-mlir — shared builder layer for all MLIR-based languages.
# Reads MLIR IR (types, ops) and drives the MLIR-free core (tile-smt) through
# the Context builder API. Holds: dtypeOf, Env (mlir::Value -> Value), the
# MemId-mapping + block-walk driver (State), and the standard-dialect handlers
# (arith / math / scf). Links the core + MLIR (deliberately NOT z3-only).
#
# NOTE: State's interpretOp dispatch references the tt.* handlers declared in
# builder/triton (handleTt*). Those symbols are provided by
# tile-smt-builder-triton; as a static archive this library needs no resolution
# until the final executable / test link, which links builder-triton.

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# Compiled WITHOUT -fno-exceptions/-fno-rtti so std::throw (Env::lookup) and the
# z3++ exceptions work normally.
add_library(tile-smt-builder-mlir STATIC
  DTypeOf.cpp
  Env.cpp
  State.cpp
  ArithOps.cpp
)

# tv/ is the include root: code uses #include "builder/mlir/State.h",
# "semantics/Context.h", etc.
target_include_directories(tile-smt-builder-mlir PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(tile-smt-builder-mlir PUBLIC
  tile-smt          # MLIR-free core (Context / Memory / Value / AbstractFp)
  ${triton_libs}    # Triton (+ arith/math dialects it depends on)
  MLIRIR
  MLIRSupport
)
