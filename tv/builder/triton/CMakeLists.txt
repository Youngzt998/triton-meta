# tile-smt-builder-triton — Triton-specific builder layer. Handlers for tt.*
# ops built on top of builder/mlir (which provides State / Env / dtypeOf and the
# arith/math/scf handlers the dispatch also calls). Links core + builder-mlir +
# Triton dialect (brought in transitively via builder-mlir's ${triton_libs}).

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_library(tile-smt-builder-triton STATIC
  TritonOps.cpp
)

target_include_directories(tile-smt-builder-triton PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(tile-smt-builder-triton PUBLIC
  tile-smt-builder-mlir   # core + MLIR + State/Env/dtypeOf + arith/math handlers
  ${triton_libs}          # Triton dialect (tt.* ops / types)
)
