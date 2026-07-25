#include "builder/mlir/Env.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "SimpleTest.h"
#include <stdexcept>
#include <z3++.h>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// makeSymbolicValue — scalar integer
//===----------------------------------------------------------------------===//

TEST(Env, MakeSymbolicScalarI1) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto i1Ty = mlir::IntegerType::get(&mlirCtx, 1);

  Value v = makeSymbolicValue(i1Ty, context, "x_i1");
  ASSERT_TRUE(std::holds_alternative<Scalar>(v));
  auto &s = std::get<Scalar>(v);
  EXPECT_EQ(s.e.get_sort().sort_kind(), Z3_BOOL_SORT);
  EXPECT_EQ(s.ty, DType::I1);
}

TEST(Env, MakeSymbolicScalarI32) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);

  Value v = makeSymbolicValue(i32Ty, context, "x_i32");
  ASSERT_TRUE(std::holds_alternative<Scalar>(v));
  EXPECT_EQ(std::get<Scalar>(v).e.get_sort().bv_size(), 32u);
}

TEST(Env, MakeSymbolicScalarF32IntegerRange) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto f32Ty = mlir::Float32Type::get(&mlirCtx);

  Value v = makeSymbolicValue(f32Ty, context, "x_f32");
  ASSERT_TRUE(std::holds_alternative<Scalar>(v));
  EXPECT_EQ(std::get<Scalar>(v).e.get_sort().bv_size(), 32u);
  EXPECT_EQ(std::get<Scalar>(v).ty, DType::F32);
}

TEST(Env, MakeSymbolicScalarF32FPA) {
  z3::context ctx;
  Context context(ctx, FPMode::FPA);
  mlir::MLIRContext mlirCtx;
  auto f32Ty = mlir::Float32Type::get(&mlirCtx);

  Value v = makeSymbolicValue(f32Ty, context, "x_f32_fpa");
  ASSERT_TRUE(std::holds_alternative<Scalar>(v));
  auto &s = std::get<Scalar>(v);
  EXPECT_EQ(s.e.get_sort().sort_kind(), Z3_FLOATING_POINT_SORT);
  EXPECT_EQ(s.ty, DType::F32);
}

//===----------------------------------------------------------------------===//
// makeSymbolicValue — tensor
//===----------------------------------------------------------------------===//

TEST(Env, MakeSymbolicTile1D) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto tensorTy = mlir::RankedTensorType::get({16}, i32Ty);

  Value v = makeSymbolicValue(tensorTy, context, "t1d");
  ASSERT_TRUE(std::holds_alternative<Tensor>(v));
  auto &t = std::get<Tensor>(v);

  EXPECT_EQ(t.shape.size(), 1u);
  EXPECT_EQ(t.shape[0], 16);
  EXPECT_EQ(t.elem, DType::I32);

  // Sort must be Array(BitVec(32), BitVec(32))
  EXPECT_EQ(t.e.get_sort().sort_kind(), Z3_ARRAY_SORT);
  EXPECT_EQ(t.e.get_sort().array_domain().bv_size(), 32u);
  EXPECT_EQ(t.e.get_sort().array_range().bv_size(), 32u);
}

TEST(Env, MakeSymbolicTile2D) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto tensorTy = mlir::RankedTensorType::get({4, 8}, f32Ty);

  Value v = makeSymbolicValue(tensorTy, context, "t2d");
  ASSERT_TRUE(std::holds_alternative<Tensor>(v));
  auto &t = std::get<Tensor>(v);

  EXPECT_EQ(t.shape.size(), 2u);
  EXPECT_EQ(t.shape[0], 4);
  EXPECT_EQ(t.shape[1], 8);
  EXPECT_EQ(t.elem, DType::F32);
}

TEST(Env, TwoTilesWithDistinctNamesAreDistinct) {
  // Two fresh tiles should be distinguishable by the solver.
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto tensorTy = mlir::RankedTensorType::get({4}, i32Ty);

  Value v1 = makeSymbolicValue(tensorTy, context, "tA");
  Value v2 = makeSymbolicValue(tensorTy, context, "tB");

  auto &t1 = std::get<Tensor>(v1);
  auto &t2 = std::get<Tensor>(v2);

  z3::solver solver(ctx);
  solver.add(t1.e != t2.e);
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// makeSymbolicValue — Triton pointer
//===----------------------------------------------------------------------===//

TEST(Env, MakeSymbolicPtr) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto f32Ty = mlir::Float32Type::get(&mlirCtx);
  auto ptrTy = mlir::triton::PointerType::get(f32Ty, 1);

  Value v = makeSymbolicValue(ptrTy, context, "p");
  ASSERT_TRUE(std::holds_alternative<Ptr>(v));
  auto &p = std::get<Ptr>(v);
  EXPECT_EQ(p.e.get_sort().bv_size(), 64u);
  EXPECT_EQ(p.pointee, DType::F32);
}

//===----------------------------------------------------------------------===//
// Env — bind / lookup / contains / size
//===----------------------------------------------------------------------===//

// We need a real mlir::Value to test Env. The simplest way is to create a
// small function with arguments via the MLIR builder.
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

static mlir::OwningOpRef<mlir::ModuleOp>
makeModuleWithFunc(mlir::MLIRContext &mlirCtx,
                   llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::OpBuilder builder(&mlirCtx);
  auto loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(loc);
  auto funcTy = builder.getFunctionType(argTypes, {});
  builder.setInsertionPointToEnd(module.getBody());
  auto func = mlir::triton::FuncOp::create(builder, loc, "test_fn", funcTy);
  func.addEntryBlock();
  return module;
}

TEST(Env, BindAndLookup) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto module = makeModuleWithFunc(mlirCtx, {i32Ty, i32Ty});

  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args = func.getBody().getArguments();

  Value v0 = makeSymbolicValue(i32Ty, context, "a0");
  Value v1 = makeSymbolicValue(i32Ty, context, "a1");

  Env env;
  EXPECT_EQ(env.size(), 0u);

  env.bind(args[0], v0);
  env.bind(args[1], v1);

  EXPECT_EQ(env.size(), 2u);
  EXPECT_TRUE(env.contains(args[0]));
  EXPECT_TRUE(env.contains(args[1]));

  // Looked-up expressions carry the right names.
  auto &r0 = std::get<Scalar>(env.lookup(args[0]));
  auto &r1 = std::get<Scalar>(env.lookup(args[1]));
  EXPECT_TRUE(r0.e.to_string().find("a0") != std::string::npos);
  EXPECT_TRUE(r1.e.to_string().find("a1") != std::string::npos);
}

TEST(Env, LookupMissingThrows) {
  z3::context ctx;
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto module = makeModuleWithFunc(mlirCtx, {i32Ty});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  Env env;
  EXPECT_THROW(env.lookup(arg), std::out_of_range);
}

TEST(Env, BindOverwrites) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto module = makeModuleWithFunc(mlirCtx, {i32Ty});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto arg = func.getBody().getArguments()[0];

  Env env;
  env.bind(arg, makeSymbolicValue(i32Ty, context, "old"));
  env.bind(arg, makeSymbolicValue(i32Ty, context, "new"));

  EXPECT_EQ(env.size(), 1u);
  auto &s = std::get<Scalar>(env.lookup(arg));
  EXPECT_TRUE(s.e.to_string().find("new") != std::string::npos);
}

//===----------------------------------------------------------------------===//
// initFuncArgs
//===----------------------------------------------------------------------===//

TEST(Env, InitFuncArgs) {
  z3::context ctx;
  Context context(ctx, FPMode::IntegerRange);
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::triton::TritonDialect>();

  auto i32Ty = mlir::IntegerType::get(&mlirCtx, 32);
  auto tensorTy = mlir::RankedTensorType::get({8}, i32Ty);
  auto module = makeModuleWithFunc(mlirCtx, {i32Ty, tensorTy});
  auto func = *module->getBody()->op_begin<mlir::triton::FuncOp>();
  auto args = func.getBody().getArguments();

  Env env;
  initFuncArgs(env, args, context, "src");

  EXPECT_EQ(env.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<Scalar>(env.lookup(args[0])));
  EXPECT_TRUE(std::holds_alternative<Tensor>(env.lookup(args[1])));

  // Verify name prefix was applied.
  auto &s = std::get<Scalar>(env.lookup(args[0]));
  EXPECT_TRUE(s.e.to_string().find("src_arg0") != std::string::npos);
}

int main() { return simpletest::runAll(); }
