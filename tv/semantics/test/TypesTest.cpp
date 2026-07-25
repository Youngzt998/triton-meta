// Z3-only unit test for the tile-smt core Types. Deliberately includes NO
// MLIR/Triton headers — this is the proof the core builds/runs without MLIR.

#include "semantics/Types.h"
#include "SimpleTest.h"

#include <z3++.h>

using namespace tile_smt;

TEST(Types, ByteWidth) {
  EXPECT_EQ(getByteWidth(DType::I1), 1u);
  EXPECT_EQ(getByteWidth(DType::I8), 1u);
  EXPECT_EQ(getByteWidth(DType::I16), 2u);
  EXPECT_EQ(getByteWidth(DType::I32), 4u);
  EXPECT_EQ(getByteWidth(DType::I64), 8u);
  EXPECT_EQ(getByteWidth(DType::F16), 2u);
  EXPECT_EQ(getByteWidth(DType::BF16), 2u);
  EXPECT_EQ(getByteWidth(DType::F32), 4u);
  EXPECT_EQ(getByteWidth(DType::F64), 8u);
  EXPECT_EQ(getByteWidth(DType::Ptr), 8u);
}

TEST(Types, FpExpSigBits) {
  EXPECT_EQ(fpExpSigBits(DType::F16).first, 5u);
  EXPECT_EQ(fpExpSigBits(DType::F16).second, 11u);
  EXPECT_EQ(fpExpSigBits(DType::BF16).first, 8u);
  EXPECT_EQ(fpExpSigBits(DType::F32).second, 24u);
  EXPECT_EQ(fpExpSigBits(DType::F64).second, 53u);
}

TEST(Types, ElemSortAbstractAndInt) {
  z3::context ctx;
  EXPECT_TRUE(getElemSort(ctx, DType::I1, FPMode::Abstract).is_bool());
  EXPECT_EQ(getElemSort(ctx, DType::I32, FPMode::Abstract).bv_size(), 32u);
  EXPECT_EQ(getElemSort(ctx, DType::I64, FPMode::Abstract).bv_size(), 64u);
  // Abstract FP: opaque BitVec of the IEEE width.
  EXPECT_EQ(getElemSort(ctx, DType::F32, FPMode::Abstract).bv_size(), 32u);
  EXPECT_EQ(getElemSort(ctx, DType::F64, FPMode::Abstract).bv_size(), 64u);
  // Pointers are 64-bit addresses.
  EXPECT_EQ(getElemSort(ctx, DType::Ptr, FPMode::Abstract).bv_size(), 64u);
}

TEST(Types, ElemSortFpaAndReal) {
  z3::context ctx;
  auto f32fpa = getElemSort(ctx, DType::F32, FPMode::FPA);
  EXPECT_EQ(f32fpa.fpa_ebits(), 8u);
  EXPECT_EQ(f32fpa.fpa_sbits(), 24u);
  EXPECT_EQ(getElemSort(ctx, DType::F32, FPMode::Real).is_real(), true);
}

int main() { return simpletest::runAll(); }
