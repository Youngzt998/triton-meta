// Z3-only unit tests for the tile-smt core checkEquivalence (final-memory
// equivalence via the lambda store + pointwise witness comparison). NO MLIR:
// memories are keyed by the neutral tile_smt::MemId and the pairing is a plain
// vector<pair<MemId,MemId>>, proving the top-level equivalence query
// builds/runs without any MLIR present.

#include "semantics/Equivalence.h"
#include "semantics/Memory.h"
#include "semantics/Value.h"

#include "SimpleTest.h"
#include <utility>
#include <vector>
#include <z3++.h>

using namespace tile_smt;

//===----------------------------------------------------------------------===//
// Test helpers (same tile builders as MemoryTest, kept local).
//===----------------------------------------------------------------------===//

static Tensor makePtrTile(z3::context &ctx, DType elem, uint64_t baseAddr,
                          uint64_t stride, Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  z3::expr base = ctx.bv_val((uint64_t)baseAddr, 64);
  z3::expr st = ctx.bv_val((uint64_t)stride, 64);
  return Tensor{z3::lambda(i, base + st * z3::zext(i, 32)), shape, elem,
                std::nullopt};
}

static Tensor makeMaskTile(z3::context &ctx, bool val, Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bool_val(val)), shape, DType::I1,
                std::nullopt};
}

static Tensor makeConstTile(z3::context &ctx, DType elem, uint32_t val,
                            Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bv_val(val, 32)), shape, elem, std::nullopt};
}

// Insert a fresh Memory into `ms` under `id` (Memory is not assignable).
static Memory &addMem(MemState &ms, z3::context &ctx, MemId id,
                      const std::string &name) {
  ms.mems.emplace(std::piecewise_construct, std::forward_as_tuple(id),
                  std::forward_as_tuple(ctx, FPMode::IntegerRange, name));
  return ms.mems.at(id);
}

//===----------------------------------------------------------------------===//
// checkEquivalence — the validator's top-level query, MLIR-free.
//===----------------------------------------------------------------------===//

TEST(Equivalence, IdenticalStoresUNSAT) {
  // Two programs make identical stores into memories that start equal → the
  // final heaps agree at every address → UNSAT (equivalent).
  z3::context ctx;
  z3::solver solver(ctx);

  MemState s1, s2;
  Memory &m1 = addMem(s1, ctx, static_cast<MemId>(0), "s1_mem0");
  Memory &m2 = addMem(s2, ctx, static_cast<MemId>(0), "s2_mem0");

  // Same initial contents (both programs share the same input array).
  solver.add(m1.array == m2.array);

  auto ptrTile = makePtrTile(ctx, DType::I32, 0x1000, 4, {4});
  auto trueMask = makeMaskTile(ctx, true, {4});
  m1.store(ptrTile, makeConstTile(ctx, DType::I32, 7, {4}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, DType::I32, 7, {4}), trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {
      {static_cast<MemId>(0), static_cast<MemId>(0)}};
  EXPECT_EQ(checkEquivalence(s1, s2, pairing, solver), z3::unsat);
}

TEST(Equivalence, DifferingStoresSAT) {
  // Same start, but the two programs store different values → SAT (a witness
  // address where the heaps disagree exists).
  z3::context ctx;
  z3::solver solver(ctx);

  MemState s1, s2;
  Memory &m1 = addMem(s1, ctx, static_cast<MemId>(0), "s1_mem0b");
  Memory &m2 = addMem(s2, ctx, static_cast<MemId>(0), "s2_mem0b");
  solver.add(m1.array == m2.array);

  auto ptrTile = makePtrTile(ctx, DType::I32, 0x2000, 4, {1});
  auto trueMask = makeMaskTile(ctx, true, {1});
  m1.store(ptrTile, makeConstTile(ctx, DType::I32, 42, {1}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, DType::I32, 43, {1}), trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {
      {static_cast<MemId>(0), static_cast<MemId>(0)}};
  EXPECT_EQ(checkEquivalence(s1, s2, pairing, solver), z3::sat);
}

TEST(Equivalence, EmptyPairingUNSAT) {
  // No paired memories → the disjunction is `false` → UNSAT (nothing to
  // differ).
  z3::context ctx;
  z3::solver solver(ctx);
  MemState s1, s2;
  std::vector<std::pair<MemId, MemId>> pairing;
  EXPECT_EQ(checkEquivalence(s1, s2, pairing, solver), z3::unsat);
}

TEST(Equivalence, PairingMapsDifferentIds) {
  // The pairing need not use equal MemIds on both sides: src id 0 ↔ tgt id 5.
  // A difference in that paired memory must still be caught (SAT).
  z3::context ctx;
  z3::solver solver(ctx);

  MemState s1, s2;
  Memory &m1 = addMem(s1, ctx, static_cast<MemId>(0), "s1_mem_a");
  Memory &m2 = addMem(s2, ctx, static_cast<MemId>(5), "s2_mem_b");
  solver.add(m1.array == m2.array);

  auto ptrTile = makePtrTile(ctx, DType::I32, 0x3000, 4, {1});
  auto trueMask = makeMaskTile(ctx, true, {1});
  m1.store(ptrTile, makeConstTile(ctx, DType::I32, 1, {1}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, DType::I32, 2, {1}), trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {
      {static_cast<MemId>(0), static_cast<MemId>(5)}};
  EXPECT_EQ(checkEquivalence(s1, s2, pairing, solver), z3::sat);
}

int main() { return simpletest::runAll(); }
