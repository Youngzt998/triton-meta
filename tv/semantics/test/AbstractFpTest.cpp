// Z3-only unit tests for the tile-smt core Abstract FP model. NO MLIR: the FP
// type is the neutral tile_smt::DType, proving AbstractFp builds/runs without
// any MLIR present.

#include "semantics/AbstractFp.h"

#include "SimpleTest.h"
#include <z3++.h>

using namespace tile_smt;

//===----------------------------------------------------------------------===//
// Sort / bitwidth
//===----------------------------------------------------------------------===//

TEST(AbstractFp, SortIsBitVecOfTypeWidth) {
  z3::context ctx;

  AbstractFp e32(ctx, DType::F32);
  AbstractFp e64(ctx, DType::F64);

  EXPECT_EQ(e32.sort().sort_kind(), Z3_BV_SORT);
  EXPECT_EQ(e32.sort().bv_size(), 32u);
  EXPECT_EQ(e64.sort().bv_size(), 64u);
}

//===----------------------------------------------------------------------===//
// Reserved constants — pairwise distinct after axioms are emitted.
//===----------------------------------------------------------------------===//

TEST(AbstractFp, ReservedConstantsAreDistinct) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  // Materialize all five reserved consts.
  z3::expr pz = e.posZero();
  z3::expr nz = e.negZero();
  z3::expr pi = e.posInf();
  z3::expr ni = e.negInf();
  z3::expr nv = e.nan();

  z3::solver solver(ctx);
  e.addAxioms(solver);

  // Each pair should be unequal (the distinct axiom enforces this).
  z3::expr_vector consts(ctx);
  consts.push_back(pz); consts.push_back(nz);
  consts.push_back(pi); consts.push_back(ni);
  consts.push_back(nv);

  for (unsigned i = 0; i < consts.size(); i++) {
    for (unsigned j = i + 1; j < consts.size(); j++) {
      solver.push();
      solver.add(consts[i] == consts[j]);
      EXPECT_EQ(solver.check(), z3::unsat);
      solver.pop();
    }
  }
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

TEST(AbstractFp, IsZeroPredicateHoldsForReservedZeros) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::solver solver(ctx);
  e.addAxioms(solver);

  solver.push();
  solver.add(!e.isZero(e.posZero()));
  EXPECT_EQ(solver.check(), z3::unsat);
  solver.pop();

  solver.push();
  solver.add(!e.isZero(e.negZero()));
  EXPECT_EQ(solver.check(), z3::unsat);
  solver.pop();

  // A fresh symbolic value should not be forced to be zero.
  z3::expr x = ctx.bv_const("x", 32);
  solver.push();
  solver.add(e.isZero(x));
  EXPECT_EQ(solver.check(), z3::sat);
  solver.pop();
}

//===----------------------------------------------------------------------===//
// Uninterpreted functions: same inputs ⇒ same outputs (functional).
//===----------------------------------------------------------------------===//

TEST(AbstractFp, AddIsFunctional) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  // No axioms required: this is the SMT semantics of an uninterpreted fn.
  solver.add(e.add(a, b) != e.add(a, b));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, DistinctArgsCanProduceDistinctResults) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);
  z3::expr c = ctx.bv_const("c", 32);

  z3::solver solver(ctx);
  e.addAxioms(solver);
  solver.add(a != c);
  solver.add(e.add(a, b) == e.add(c, b));
  // The solver is free to pick a model where add happens to collide; allowed.
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// Axioms
//===----------------------------------------------------------------------===//

TEST(AbstractFp, AddCommutativityAxiom) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  // Touch add so the axiom is emitted.
  (void)e.add(a, b);
  e.addAxioms(solver);

  solver.add(e.add(a, b) != e.add(b, a));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, MulCommutativityAxiom) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  (void)e.mul(a, b);
  e.addAxioms(solver);

  solver.add(e.mul(a, b) != e.mul(b, a));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, NegIsInvolutive) {
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr x = ctx.bv_const("x", 32);

  z3::solver solver(ctx);
  (void)e.neg(x);
  e.addAxioms(solver);

  solver.add(e.neg(e.neg(x)) != x);
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, AddIsNotTriviallyAssociative) {
  // Sanity check that the encoding does NOT silently impose associativity
  // (mlir-tv treats associativity as opt-in; we follow the same default).
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);
  z3::expr c = ctx.bv_const("c", 32);

  z3::solver solver(ctx);
  (void)e.add(a, b);
  e.addAxioms(solver);

  solver.add(e.add(e.add(a, b), c) != e.add(a, e.add(b, c)));
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// Per-type independence
//===----------------------------------------------------------------------===//

TEST(AbstractFp, F32AndF64UseSeparateFunctions) {
  z3::context ctx;

  AbstractFp e32(ctx, DType::F32);
  AbstractFp e64(ctx, DType::F64);

  // Their add functions are distinct — different domain widths — so they
  // simply cannot be combined; this is a smoke test that both work.
  z3::expr a32 = ctx.bv_const("a32", 32), b32 = ctx.bv_const("b32", 32);
  z3::expr a64 = ctx.bv_const("a64", 64), b64 = ctx.bv_const("b64", 64);

  EXPECT_EQ(e32.add(a32, b32).get_sort().bv_size(), 32u);
  EXPECT_EQ(e64.add(a64, b64).get_sort().bv_size(), 64u);
}

//===----------------------------------------------------------------------===//
// Registry
//===----------------------------------------------------------------------===//

TEST(AbstractFp, RegistryReturnsSameInstanceForSameType) {
  z3::context ctx;
  AbstractFpRegistry reg(ctx);

  AbstractFp &a = reg.get(DType::F32);
  AbstractFp &b = reg.get(DType::F32);
  EXPECT_EQ(&a, &b);
}

TEST(AbstractFp, RegistryDistinguishesF16AndBF16) {
  z3::context ctx;
  AbstractFpRegistry reg(ctx);

  AbstractFp &e16  = reg.get(DType::F16);
  AbstractFp &eb16 = reg.get(DType::BF16);
  EXPECT_NE(&e16, &eb16);
  EXPECT_EQ(e16.bitwidth(),  16u);
  EXPECT_EQ(eb16.bitwidth(), 16u);
}

//===----------------------------------------------------------------------===//
// max / exp (added for softmax: arith.maxnumf, math.exp)
//===----------------------------------------------------------------------===//

TEST(AbstractFp, MaxAndExpAreFunctional) {
  // Same inputs ⇒ same outputs (uninterpreted-function semantics).
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  solver.add((e.max(a, b) != e.max(a, b)) || (e.exp(a) != e.exp(a)));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, MaxCommutativityAxiom) {
  // maxnumf is commutative; the axiom must make max(a,b) == max(b,a).
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  (void)e.max(a, b);   // touch max so the axiom is emitted
  e.addAxioms(solver);

  solver.add(e.max(a, b) != e.max(b, a));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(AbstractFp, ExpIsNotTriviallyConstant) {
  // exp is uninterpreted: distinct inputs may map to distinct outputs (the
  // solver is free to pick such a model — exp is not forced to be constant).
  z3::context ctx;
  AbstractFp e(ctx, DType::F32);

  z3::expr a = ctx.bv_const("a", 32);
  z3::expr b = ctx.bv_const("b", 32);

  z3::solver solver(ctx);
  e.addAxioms(solver);
  solver.add(a != b);
  solver.add(e.exp(a) != e.exp(b));
  EXPECT_EQ(solver.check(), z3::sat);
}

int main() { return simpletest::runAll(); }
