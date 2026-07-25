#include "semantics/mlir/ArithOps.h"

#include "semantics/mlir/AbstractFpShim.h"
#include "semantics/Memory.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>

using namespace Semantics;

using tile_smt::getElemSort;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Build a Tensor from a uniform (splat) constant value.
static Tensor splatConstTile(z3::context &ctx, z3::expr val, DType elem,
                             llvm::ArrayRef<int64_t> shape) {
  z3::expr i = ctx.bv_const("__ci", 32);
  tile_smt::Shape shapeVec(shape.begin(), shape.end());
  return Tensor{z3::lambda(i, val), std::move(shapeVec), elem, std::nullopt};
}

// Apply a binary operation element-wise to two Values (scalar or tile).
// op_fn takes (lhs_elem, rhs_elem) and returns the result element.
// resultElem describes the result element type (may differ from input, e.g.
// cmpi returns i1 from integer inputs).
static Value applyBinaryOp(
    z3::context &ctx,
    const Value &lhs, const Value &rhs, DType resultElem,
    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {

  if (auto *l = std::get_if<Scalar>(&lhs)) {
    z3::expr res = op_fn(l->e, std::get<Scalar>(rhs).e);
    return Scalar{res, resultElem};
  }
  if (auto *l = std::get_if<Tensor>(&lhs)) {
    auto *r = &std::get<Tensor>(rhs);
    z3::expr i = ctx.bv_const("__bi", 32);
    z3::expr res = op_fn(z3::select(l->e, i), z3::select(r->e, i));
    return Tensor{z3::lambda(i, res), l->shape, resultElem, std::nullopt};
  }
  llvm_unreachable("applyBinaryOp: unexpected Value variant");
}

// Overload that preserves the input element type.
static Value applyBinaryOp(
    z3::context &ctx, const Value &lhs, const Value &rhs,
    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {
  DType ty = std::holds_alternative<Scalar>(lhs)
                 ? std::get<Scalar>(lhs).ty
                 : std::get<Tensor>(lhs).elem;
  return applyBinaryOp(ctx, lhs, rhs, ty, op_fn);
}

// Get the element DType of an op result (scalar or tensor).
static DType getResultElemType(mlir::Operation *op) {
  mlir::Type t = op->getResult(0).getType();
  if (auto tt = llvm::dyn_cast<mlir::RankedTensorType>(t))
    return dtypeOf(tt.getElementType());
  return dtypeOf(t);
}

//===----------------------------------------------------------------------===//
// arith.constant
//===----------------------------------------------------------------------===//

State Semantics::handleArithConstant(const State &s, mlir::Operation *op) {
  auto cop     = llvm::cast<mlir::arith::ConstantOp>(op);
  mlir::Type   resType = op->getResult(0).getType();
  mlir::Attribute value = cop.getValue();
  z3::context &ctx = s.ctx;

  // Compute the Value through a lambda so all branches return directly,
  // avoiding any need to default-construct Value (z3::expr has no default ctor).
  auto makeVal = [&]() -> Value {
    // --- scalar integer ---
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
      auto intTy = llvm::cast<mlir::IntegerType>(resType);
      unsigned bw = intTy.getWidth();
      DType elem = dtypeOf(resType);
      if (bw == 1)
        return Scalar{ctx.bool_val(intAttr.getInt() != 0), elem};
      return Scalar{ctx.bv_val(intAttr.getInt(), bw), elem};
    }
    // --- dense integer tensor ---
    if (auto denseAttr = llvm::dyn_cast<mlir::DenseIntElementsAttr>(value)) {
      auto tensorTy = llvm::cast<mlir::RankedTensorType>(resType);
      mlir::Type elemTy = tensorTy.getElementType();
      DType elem = dtypeOf(elemTy);
      unsigned bw = llvm::cast<mlir::IntegerType>(elemTy).getWidth();
      llvm::SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                                       tensorTy.getShape().end());
      if (denseAttr.isSplat()) {
        int64_t v = denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
        z3::expr val = (bw == 1) ? ctx.bool_val(v != 0) : ctx.bv_val(v, bw);
        return splatConstTile(ctx, val, elem, shape);
      }
      // Non-splat: build via z3::store chain.
      z3::sort elemSort = getElemSort(ctx, elem, s.fpMode);
      z3::expr arr = ctx.constant("__dense_base",
                                  ctx.array_sort(ctx.bv_sort(32), elemSort));
      unsigned idx = 0;
      for (const llvm::APInt &apv : denseAttr) {
        int64_t v  = apv.getSExtValue();
        z3::expr e = (bw == 1) ? ctx.bool_val(v != 0) : ctx.bv_val(v, bw);
        arr = z3::store(arr, ctx.bv_val(idx, 32), e);
        ++idx;
      }
      tile_smt::Shape shapeVec(shape.begin(), shape.end());
      return Tensor{arr, std::move(shapeVec), elem, std::nullopt};
    }
    // --- dense float tensor ---
    if (auto denseFP = llvm::dyn_cast<mlir::DenseFPElementsAttr>(value)) {
      auto tensorTy = llvm::cast<mlir::RankedTensorType>(resType);
      auto floatTy  = llvm::cast<mlir::FloatType>(tensorTy.getElementType());
      DType elem    = dtypeOf(floatTy);
      unsigned bw   = floatTy.getWidth();
      llvm::SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                                       tensorTy.getShape().end());
      // Abstract/IntegerRange carry FP as BitVec(width); use the IEEE bit
      // pattern as the opaque id (same convention as the scalar float case).
      if (denseFP.isSplat()) {
        uint64_t bits =
            denseFP.getSplatValue<llvm::APFloat>().bitcastToAPInt().getZExtValue();
        return splatConstTile(ctx, ctx.bv_val(bits, bw), elem, shape);
      }
      z3::sort elemSort = getElemSort(ctx, elem, s.fpMode);
      z3::expr arr = ctx.constant("__dense_fp_base",
                                  ctx.array_sort(ctx.bv_sort(32), elemSort));
      unsigned idx = 0;
      for (const llvm::APFloat &apf : denseFP.getValues<llvm::APFloat>()) {
        uint64_t bits = apf.bitcastToAPInt().getZExtValue();
        arr = z3::store(arr, ctx.bv_val(idx, 32), ctx.bv_val(bits, bw));
        ++idx;
      }
      tile_smt::Shape shapeVec(shape.begin(), shape.end());
      return Tensor{arr, std::move(shapeVec), elem, std::nullopt};
    }
    // --- scalar float ---
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(value)) {
      auto floatTy = llvm::cast<mlir::FloatType>(resType);
      unsigned bw  = floatTy.getWidth();
      uint64_t bits = floatAttr.getValue().bitcastToAPInt().getZExtValue();
      return Scalar{ctx.bv_val(bits, bw), dtypeOf(floatTy)};
    }
    llvm_unreachable("arith.constant: unsupported attribute kind");
  };

  State next = s;
  next.env.bind(op->getResult(0), makeVal());
  return next;
}

//===----------------------------------------------------------------------===//
// Integer binary ops — arith.addi, subi, muli, andi
//===----------------------------------------------------------------------===//

State Semantics::handleArithAddi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a + b; }));
  return next;
}

State Semantics::handleArithSubi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a - b; }));
  return next;
}

State Semantics::handleArithMuli(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [](z3::expr a, z3::expr b) { return a * b; }));
  return next;
}

State Semantics::handleArithAndi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  DType elem = getResultElemType(op);

  // i1 → Bool in Z3: use logical &&. Other widths: use bitwise &.
  bool isBool = (elem == DType::I1);

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs,
      [isBool](z3::expr a, z3::expr b) -> z3::expr {
        return isBool ? (a && b) : (a & b);
      }));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.extsi — sign-extend
//===----------------------------------------------------------------------===//

State Semantics::handleArithExtsi(const State &s, mlir::Operation *op) {
  auto extOp = llvm::cast<mlir::arith::ExtSIOp>(op);
  Value src = s.env.lookup(extOp.getIn());
  mlir::Type dstType = extOp.getType();
  z3::context &ctx = s.ctx;

  // Determine destination bit width.
  auto getDstBw = [&](mlir::Type t) -> unsigned {
    if (auto tt = llvm::dyn_cast<mlir::RankedTensorType>(t))
      return llvm::cast<mlir::IntegerType>(tt.getElementType()).getWidth();
    return llvm::cast<mlir::IntegerType>(t).getWidth();
  };
  unsigned dstBw = getDstBw(dstType);

  auto extend = [&](const Value &v, DType dstElem) -> Value {
    if (auto *sc = std::get_if<Scalar>(&v)) {
      unsigned srcBw = sc->e.get_sort().bv_size();
      return Scalar{z3::sext(sc->e, dstBw - srcBw), dstElem};
    }
    if (auto *ti = std::get_if<Tensor>(&v)) {
      unsigned srcBw = ti->e.get_sort().array_range().bv_size();
      unsigned extra = dstBw - srcBw;
      z3::expr i     = ctx.bv_const("__ei", 32);
      z3::expr elem  = z3::sext(z3::select(ti->e, i), extra);
      return Tensor{z3::lambda(i, elem), ti->shape, dstElem, std::nullopt};
    }
    llvm_unreachable("extsi: unexpected Value variant");
  };

  DType dstElem =
      llvm::isa<mlir::RankedTensorType>(dstType)
          ? dtypeOf(llvm::cast<mlir::RankedTensorType>(dstType).getElementType())
          : dtypeOf(dstType);

  State next = s;
  next.env.bind(op->getResult(0), extend(src, dstElem));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.cmpi — integer comparison
//===----------------------------------------------------------------------===//

State Semantics::handleArithCmpi(const State &s, mlir::Operation *op) {
  auto cmpiOp = llvm::cast<mlir::arith::CmpIOp>(op);
  Value lhs = s.env.lookup(cmpiOp.getLhs());
  Value rhs = s.env.lookup(cmpiOp.getRhs());
  mlir::arith::CmpIPredicate pred = cmpiOp.getPredicate();

  // Result element type is always i1 (Bool in Z3).
  auto cmpFn = [pred](z3::expr a, z3::expr b) -> z3::expr {
    switch (pred) {
    case mlir::arith::CmpIPredicate::eq:  return a == b;
    case mlir::arith::CmpIPredicate::ne:  return a != b;
    case mlir::arith::CmpIPredicate::slt: return z3::slt(a, b);
    case mlir::arith::CmpIPredicate::sle: return z3::sle(a, b);
    case mlir::arith::CmpIPredicate::sgt: return z3::sgt(a, b);
    case mlir::arith::CmpIPredicate::sge: return z3::sge(a, b);
    case mlir::arith::CmpIPredicate::ult: return z3::ult(a, b);
    case mlir::arith::CmpIPredicate::ule: return z3::ule(a, b);
    case mlir::arith::CmpIPredicate::ugt: return z3::ugt(a, b);
    case mlir::arith::CmpIPredicate::uge: return z3::uge(a, b);
    }
    llvm_unreachable("unhandled CmpIPredicate");
  };

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs, DType::I1, cmpFn));
  return next;
}

//===----------------------------------------------------------------------===//
// FP binary ops — arith.addf, subf, mulf
//===----------------------------------------------------------------------===//

// Generic handler for element-wise FP binary ops.
static State handleFpBinaryOp(
    const State &s, mlir::Operation *op,
    const std::function<z3::expr(AbstractFp &, z3::expr, z3::expr)> &op_fn) {

  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));

  // Get float type from the result.
  mlir::Type resType = op->getResult(0).getType();
  mlir::Type elemTy  = llvm::isa<mlir::RankedTensorType>(resType)
                           ? llvm::cast<mlir::RankedTensorType>(resType).getElementType()
                           : resType;
  auto floatTy = llvm::cast<mlir::FloatType>(elemTy);
  AbstractFp &afp = getFp(*s.fpReg, floatTy);

  auto fn = [&](z3::expr a, z3::expr b) { return op_fn(afp, a, b); };

  State next = s;
  next.env.bind(op->getResult(0),
    applyBinaryOp(s.ctx, lhs, rhs, dtypeOf(floatTy), fn));
  return next;
}

State Semantics::handleArithAddf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.add(a, b); });
}

State Semantics::handleArithSubf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.sub(a, b); });
}

State Semantics::handleArithMulf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.mul(a, b); });
}

State Semantics::handleArithDivf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.div(a, b); });
}

State Semantics::handleArithMaxnumf(const State &s, mlir::Operation *op) {
  return handleFpBinaryOp(s, op,
    [](AbstractFp &afp, z3::expr a, z3::expr b) { return afp.max(a, b); });
}

//===----------------------------------------------------------------------===//
// math.exp — element-wise FP unary (uninterpreted in Abstract mode)
//===----------------------------------------------------------------------===//

State Semantics::handleMathExp(const State &s, mlir::Operation *op) {
  Value in = s.env.lookup(op->getOperand(0));
  mlir::Type resType = op->getResult(0).getType();
  mlir::Type elemTy  = llvm::isa<mlir::RankedTensorType>(resType)
                           ? llvm::cast<mlir::RankedTensorType>(resType).getElementType()
                           : resType;
  auto floatTy = llvm::cast<mlir::FloatType>(elemTy);
  AbstractFp &afp = getFp(*s.fpReg, floatTy);
  DType elem = dtypeOf(floatTy);
  z3::context &ctx = s.ctx;

  State next = s;
  if (auto *sc = std::get_if<Scalar>(&in)) {
    next.env.bind(op->getResult(0), Scalar{afp.exp(sc->e), elem});
  } else {
    auto &t = std::get<Tensor>(in);
    z3::expr i = ctx.bv_const("__ue", 32);
    z3::expr e = afp.exp(z3::select(t.e, i));
    next.env.bind(op->getResult(0),
                  Tensor{z3::lambda(i, e), t.shape, elem, std::nullopt});
  }
  return next;
}
