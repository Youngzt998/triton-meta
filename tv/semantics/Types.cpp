#include "semantics/Types.h"

#include <stdexcept>

// Core must stay free of MLIR/LLVM: use std exceptions instead of
// llvm_unreachable for the unreachable/invalid cases.

namespace tile_smt {

unsigned getByteWidth(DType ty) {
  switch (ty) {
  case DType::I1:   return 1;
  case DType::I8:   return 1;
  case DType::I16:  return 2;
  case DType::I32:  return 4;
  case DType::I64:  return 8;
  case DType::F16:  return 2;
  case DType::BF16: return 2;
  case DType::F32:  return 4;
  case DType::F64:  return 8;
  case DType::Ptr:  return 8;
  }
  throw std::logic_error("getByteWidth: unhandled DType");
}

std::pair<unsigned, unsigned> fpExpSigBits(DType ty) {
  switch (ty) {
  case DType::F16:  return {5, 11};
  case DType::BF16: return {8, 8};
  case DType::F32:  return {8, 24};
  case DType::F64:  return {11, 53};
  default:          throw std::logic_error("fpExpSigBits: not a float DType");
  }
}

z3::sort getElemSort(z3::context &ctx, DType ty, FPMode fpMode) {
  switch (ty) {
  case DType::I1:  return ctx.bool_sort();
  case DType::I8:  return ctx.bv_sort(8);
  case DType::I16: return ctx.bv_sort(16);
  case DType::I32: return ctx.bv_sort(32);
  case DType::I64: return ctx.bv_sort(64);
  case DType::F16:
  case DType::BF16:
  case DType::F32:
  case DType::F64: {
    unsigned width = getByteWidth(ty) * 8;
    switch (fpMode) {
    // Abstract: opaque BitVec(width) id. IntegerRange: significand bitvector.
    case FPMode::Abstract:
    case FPMode::IntegerRange:
      return ctx.bv_sort(width);
    case FPMode::Real:
      return ctx.real_sort();
    case FPMode::FPA: {
      auto [expBits, sigBits] = fpExpSigBits(ty);
      return ctx.fpa_sort(expBits, sigBits);
    }
    }
    throw std::logic_error("getElemSort: unhandled FPMode");
  }
  case DType::Ptr:
    return ctx.bv_sort(64);
  }
  throw std::logic_error("getElemSort: unhandled DType");
}

} // namespace tile_smt
