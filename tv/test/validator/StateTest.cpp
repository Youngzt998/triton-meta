#include "builder/mlir/State.h"
#include "semantics/Equivalence.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "SimpleTest.h"
#include <utility>
#include <vector>
#include <z3++.h>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Build a module with a single function whose arguments are `argTypes`.
// Used to obtain real mlir::Value objects (function arguments) as map keys.
static mlir::OwningOpRef<mlir::ModuleOp>
makeModule(mlir::MLIRContext &ctx, llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(loc);
  auto funcTy = builder.getFunctionType(argTypes, {});
  builder.setInsertionPointToEnd(module.getBody());
  auto func = mlir::triton::FuncOp::create(builder, loc, "test_fn", funcTy);
  func.addEntryBlock();
  return module;
}

// A pointer tile: ptr[i] = base + i*stride. ptrBase is left empty — these tests
// call Memory::store/load directly (which does not read ptrBase); provenance is
// exercised separately by the handlers.
static Tensor makePtrTile(z3::context &ctx, DType elem, uint64_t base,
                          uint64_t stride, tile_smt::Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  z3::expr b = ctx.bv_val(base, 64);
  z3::expr st = ctx.bv_val(stride, 64);
  return Tensor{z3::lambda(i, b + st * z3::zext(i, 32)), shape, elem,
                std::nullopt};
}

static Tensor makeMaskTile(z3::context &ctx, bool val, tile_smt::Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bool_val(val)), shape, DType::I1,
                std::nullopt};
}

static Tensor makeConstTile(z3::context &ctx, DType elem, uint32_t val,
                            tile_smt::Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bv_val(val, 32)), shape, elem, std::nullopt};
}

//===----------------------------------------------------------------------===//
// initFromFunc
//===----------------------------------------------------------------------===//

TEST(State, InitFromFuncCreatesPerArgMemories) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);
  auto tenTy = mlir::RankedTensorType::get({16}, i32Ty);

  // Function: (ptr, ptr, tensor)
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy, tenTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args = func.getBody().getArguments();

  State s = State::initFromFunc(args, context, "src");

  // Two pointer args → two independent memories.
  EXPECT_EQ(s.memState.mems.size(), 2u);
  EXPECT_TRUE(s.ptrArgToMem.count(args[0]));
  EXPECT_TRUE(s.ptrArgToMem.count(args[1]));
  // Tensor arg → no memory (lives in env only).
  EXPECT_FALSE(s.ptrArgToMem.count(args[2]));

  // Env has all three args.
  EXPECT_EQ(s.env.size(), 3u);
  EXPECT_TRUE(std::holds_alternative<Ptr>(s.env.lookup(args[0])));
  EXPECT_TRUE(std::holds_alternative<Ptr>(s.env.lookup(args[1])));
  EXPECT_TRUE(std::holds_alternative<Tensor>(s.env.lookup(args[2])));
}

TEST(State, PointerArgsGetIndependentMemories) {
  // Two pointer args must have distinct (SAT for ≠) symbolic memories.
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy =
      mlir::triton::PointerType::get(mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args = func.getBody().getArguments();

  State s = State::initFromFunc(args, context, "src");

  auto &mem0 = s.memState.mems.at(s.ptrArgToMem.at(args[0]));
  auto &mem1 = s.memState.mems.at(s.ptrArgToMem.at(args[1]));

  z3::solver solver(ctx);
  solver.add(mem0.array != mem1.array);
  EXPECT_EQ(solver.check(), z3::sat)
      << "each pointer arg must have its own independent memory";
}

TEST(State, PtrBaseProvenanceSetOnPointerArgs) {
  // The Ptr bound for a pointer arg must carry the MemId of that arg's memory.
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy =
      mlir::triton::PointerType::get(mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  State s = State::initFromFunc({arg}, context, "src");

  auto &ptr = std::get<Ptr>(s.env.lookup(arg));
  EXPECT_EQ(ptr.base, s.ptrArgToMem.at(arg));
  // And that MemId indexes a real Memory.
  EXPECT_TRUE(s.memState.mems.count(ptr.base));
}

//===----------------------------------------------------------------------===//
// interpretBlock — empty block is a no-op
//===----------------------------------------------------------------------===//

TEST(State, InterpretEmptyBlockUnchanged) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto ptrTy =
      mlir::triton::PointerType::get(mlir::Float32Type::get(&mlirCtx), 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  State s0 = State::initFromFunc({arg}, context, "s");

  mlir::Block emptyBlock;
  State s1 = s0.interpretBlock(emptyBlock);

  // Memory for the pointer arg must be unchanged.
  MemId id = s0.ptrArgToMem.at(arg);
  z3::solver solver(ctx);
  solver.add(s0.memState.mems.at(id).array != s1.memState.mems.at(id).array);
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// checkEquivalence
//===----------------------------------------------------------------------===//

TEST(State, EquivalentStatesUNSAT) {
  // Two programs making identical stores to the same pointer arg → UNSAT.
  z3::context ctx;
  Context srcCtx(ctx, FPMode::IntegerRange);
  Context tgtCtx(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  // Both programs start from the same initial state.
  State s1 = State::initFromFunc({arg}, srcCtx, "src");
  State s2 = State::initFromFunc({arg}, tgtCtx, "tgt");

  MemId id = s1.ptrArgToMem.at(arg); // same MemId in both (minted in arg order)

  // Assert initial memories are equal (shared symbolic inputs).
  z3::solver solver(ctx);
  solver.add(s1.memState.mems.at(id).array == s2.memState.mems.at(id).array);

  // Both programs store the same value.
  auto ptrTile = makePtrTile(ctx, DType::I32, 0x1000, 4, {4});
  auto valTile = makeConstTile(ctx, DType::I32, 42, {4});
  auto trueMask = makeMaskTile(ctx, true, {4});

  s1.memState.mems.at(id).store(ptrTile, valTile, trueMask);
  s2.memState.mems.at(id).store(ptrTile, valTile, trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {{id, id}};
  EXPECT_EQ(
      tile_smt::checkEquivalence(s1.memState, s2.memState, pairing, solver),
      z3::unsat);
}

TEST(State, NonEquivalentStatesSAT) {
  // Two programs storing different values → SAT.
  z3::context ctx;
  Context srcCtx(ctx, FPMode::IntegerRange);
  Context tgtCtx(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  State s1 = State::initFromFunc({arg}, srcCtx, "src");
  State s2 = State::initFromFunc({arg}, tgtCtx, "tgt");

  MemId id = s1.ptrArgToMem.at(arg);

  z3::solver solver(ctx);
  solver.add(s1.memState.mems.at(id).array == s2.memState.mems.at(id).array);

  auto ptrTile = makePtrTile(ctx, DType::I32, 0x2000, 4, {1});
  auto trueMask = makeMaskTile(ctx, true, {1});

  s1.memState.mems.at(id).store(
      ptrTile, makeConstTile(ctx, DType::I32, 10, {1}), trueMask);
  s2.memState.mems.at(id).store(
      ptrTile, makeConstTile(ctx, DType::I32, 20, {1}), trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {{id, id}};
  EXPECT_EQ(
      tile_smt::checkEquivalence(s1.memState, s2.memState, pairing, solver),
      z3::sat);
}

TEST(State, TwoOutputArgsCheckedTogether) {
  // Both output memories must match. If one differs → SAT.
  z3::context ctx;
  Context srcCtx(ctx, FPMode::IntegerRange);
  Context tgtCtx(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);
  auto module = makeModule(mlirCtx, {ptrTy, ptrTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args = func.getBody().getArguments();
  auto argA = args[0];
  auto argB = args[1];

  State s1 = State::initFromFunc(args, srcCtx, "src");
  State s2 = State::initFromFunc(args, tgtCtx, "tgt");

  MemId idA = s1.ptrArgToMem.at(argA);
  MemId idB = s1.ptrArgToMem.at(argB);

  z3::solver solver(ctx);
  // Shared inputs.
  solver.add(s1.memState.mems.at(idA).array == s2.memState.mems.at(idA).array);
  solver.add(s1.memState.mems.at(idB).array == s2.memState.mems.at(idB).array);

  auto trueMask = makeMaskTile(ctx, true, {1});
  auto ptrA = makePtrTile(ctx, DType::I32, 0x1000, 4, {1});
  auto ptrB = makePtrTile(ctx, DType::I32, 0x2000, 4, {1});

  // Both programs store same value to argA.
  s1.memState.mems.at(idA).store(ptrA, makeConstTile(ctx, DType::I32, 7, {1}),
                                 trueMask);
  s2.memState.mems.at(idA).store(ptrA, makeConstTile(ctx, DType::I32, 7, {1}),
                                 trueMask);

  // Programs differ on argB.
  s1.memState.mems.at(idB).store(ptrB, makeConstTile(ctx, DType::I32, 1, {1}),
                                 trueMask);
  s2.memState.mems.at(idB).store(ptrB, makeConstTile(ctx, DType::I32, 2, {1}),
                                 trueMask);

  std::vector<std::pair<MemId, MemId>> pairing = {{idA, idA}, {idB, idB}};
  EXPECT_EQ(
      tile_smt::checkEquivalence(s1.memState, s2.memState, pairing, solver),
      z3::sat)
      << "difference in argB must be detected";
}

int main() { return simpletest::runAll(); }
