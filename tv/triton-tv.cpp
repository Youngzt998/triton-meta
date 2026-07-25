#include "../bin/RegisterTritonDialects.h"

#include "semantics/State.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <z3++.h>
#include <chrono>

static llvm::cl::opt<std::string> inputFile1(
    llvm::cl::Positional, llvm::cl::desc("<first input mlir file>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> inputFile2(
    llvm::cl::Positional, llvm::cl::desc("<second input mlir file>"),
    llvm::cl::Required);

/// Parse an MLIR file and return the module.
static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile(
    const std::string &filename, mlir::MLIRContext &context) {
  std::string errorMessage;
  auto file = mlir::openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening file '" << filename << "': " << errorMessage
                 << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

void floatingSATTest(){
  z3::context ctx;
  z3::sort fp32 = ctx.fpa_sort(8, 24); // IEEE 754 float: 8 exp bits, 24 sig bits (including hidden bit)
  z3::expr a = ctx.constant("a", fp32);
  z3::expr b = ctx.constant("b", fp32);
  z3::expr rm = ctx.fpa_rounding_mode(); // RNE (round nearest, ties to even)

  z3::expr lhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, a, b));  // a + b
  z3::expr rhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, b, a));  // b + a

  // Check if there exists a,b where a+b != b+a
  z3::solver s(ctx);
  s.add(lhs != rhs);

  auto start = std::chrono::high_resolution_clock::now();
  auto result = s.check();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - start);
  llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

  if (result == z3::unsat) {
    llvm::outs() << "Verified: a + b == b + a for all float values\n";
  } else {
    z3::model m = s.get_model();
    llvm::outs() << "Counterexample found:\n";
    llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
    llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    llvm::outs() << "  a+b = " << m.eval(lhs).to_string() << "\n";
    llvm::outs() << "  b+a = " << m.eval(rhs).to_string() << "\n";
  }

  // Check commutativity of floating-point multiplication: a*b == b*a
  llvm::outs() << "\n--- FP Multiplication commutativity ---\n";
  {
    z3::context ctx2;
    z3::sort fp32_2 = ctx2.fpa_sort(8, 24);
    z3::expr a2 = ctx2.constant("a", fp32_2);
    z3::expr b2 = ctx2.constant("b", fp32_2);
    z3::expr rm2 = ctx2.fpa_rounding_mode();

    z3::expr mulLhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, a2, b2));  // a * b
    z3::expr mulRhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, b2, a2));  // b * a

    z3::solver s2(ctx2);
    s2.add(mulLhs != mulRhs);

    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = s2.check();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = std::chrono::duration<double>(end2 - start2);
    llvm::outs() << "Solver time: " << elapsed2.count() << " s\n";

    if (result2 == z3::unsat) {
      llvm::outs() << "Verified: a * b == b * a for all float values\n";
    } else {
      z3::model m2 = s2.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m2.eval(a2).to_string() << "\n";
      llvm::outs() << "  b = " << m2.eval(b2).to_string() << "\n";
      llvm::outs() << "  a*b = " << m2.eval(mulLhs).to_string() << "\n";
      llvm::outs() << "  b*a = " << m2.eval(mulRhs).to_string() << "\n";
    }
  }
}

void realSATTest(){
  // Check commutativity of real addition: a+b == b+a
  llvm::outs() << "\n=== Real Number Tests ===\n";
  llvm::outs() << "--- Real Addition commutativity ---\n";
  {
    z3::context ctx;
    z3::expr a = ctx.real_const("a");
    z3::expr b = ctx.real_const("b");

    z3::solver s(ctx);
    s.add(a + b != b + a);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = s.check();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start);
    llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

    if (result == z3::unsat) {
      llvm::outs() << "Verified: a + b == b + a for all real values\n";
    } else {
      z3::model m = s.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
      llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    }
  }

  // Check commutativity of real multiplication: a*b == b*a
  llvm::outs() << "\n--- Real Multiplication commutativity ---\n";
  {
    z3::context ctx;
    z3::expr a = ctx.real_const("a");
    z3::expr b = ctx.real_const("b");

    z3::solver s(ctx);
    s.add(a * b != b * a);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = s.check();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start);
    llvm::outs() << "Solver time: " << elapsed.count() << " s\n";

    if (result == z3::unsat) {
      llvm::outs() << "Verified: a * b == b * a for all real values\n";
    } else {
      z3::model m = s.get_model();
      llvm::outs() << "Counterexample found:\n";
      llvm::outs() << "  a = " << m.eval(a).to_string() << "\n";
      llvm::outs() << "  b = " << m.eval(b).to_string() << "\n";
    }
  }
}

// Extract the first tt.func from a module. Returns null if none found.
static mlir::triton::FuncOp extractFunc(mlir::ModuleOp mod) {
  mlir::triton::FuncOp result;
  mod->walk([&](mlir::triton::FuncOp f) {
    result = f;
    return mlir::WalkResult::interrupt();
  });
  return result;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Translation Validator\n");

  mlir::DialectRegistry registry;
  registerTritonDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module1 = parseMLIRFile(inputFile1, context);
  if (!module1) {
    llvm::errs() << "Failed to parse: " << inputFile1 << "\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module2 = parseMLIRFile(inputFile2, context);
  if (!module2) {
    llvm::errs() << "Failed to parse: " << inputFile2 << "\n";
    return 1;
  }

  mlir::triton::FuncOp func1 = extractFunc(*module1);
  mlir::triton::FuncOp func2 = extractFunc(*module2);
  if (!func1 || !func2) {
    llvm::errs() << "Could not find tt.func in one of the modules.\n";
    return 1;
  }

  llvm::outs() << "Validating: " << func1.getName() << " vs "
               << func2.getName() << "\n";

  //--------------------------------------------------------------------------
  // Build symbolic initial states.
  //--------------------------------------------------------------------------
  z3::context ctx;
  z3::solver  solver(ctx);

  auto srcArgs = func1.getBody().getArguments();
  auto tgtArgs = func2.getBody().getArguments();

  if (srcArgs.size() != tgtArgs.size()) {
    llvm::errs() << "Argument count mismatch.\n";
    return 1;
  }

  Semantics::State s1 = Semantics::State::initFromFunc(
      srcArgs, ctx, Semantics::FPMode::Abstract, "src");
  Semantics::State s2 = Semantics::State::initFromFunc(
      tgtArgs, ctx, Semantics::FPMode::Abstract, "tgt");

  //--------------------------------------------------------------------------
  // Assert shared inputs: corresponding arguments have the same values.
  //--------------------------------------------------------------------------
  for (unsigned i = 0; i < srcArgs.size(); ++i) {
    mlir::Value sa = srcArgs[i];
    mlir::Value ta = tgtArgs[i];

    auto srcMemIt = s1.ptrArgToMem.find(sa);
    auto tgtMemIt = s2.ptrArgToMem.find(ta);
    bool srcHasMem = srcMemIt != s1.ptrArgToMem.end();
    bool tgtHasMem = tgtMemIt != s2.ptrArgToMem.end();

    if (srcHasMem && tgtHasMem) {
      // Pointer arg: initial heap contents are equal AND the pointer addresses
      // themselves must match (they're the same kernel-level argument).
      solver.add(s1.memState.mems.at(srcMemIt->second).array ==
                 s2.memState.mems.at(tgtMemIt->second).array);
      auto &sp = std::get<Semantics::Ptr>(s1.env.lookup(sa));
      auto &tp = std::get<Semantics::Ptr>(s2.env.lookup(ta));
      solver.add(sp.e == tp.e);
    } else {
      // Scalar/tensor arg: symbolic values are equal.
      std::visit([&](auto &v1) {
        using T = std::decay_t<decltype(v1)>;
        solver.add(v1.e == std::get<T>(s2.env.lookup(ta)).e);
      }, s1.env.lookup(sa));
    }
  }

  //--------------------------------------------------------------------------
  // Symbolically interpret both programs.
  //--------------------------------------------------------------------------
  auto t0 = std::chrono::high_resolution_clock::now();

  Semantics::State s1f = s1.interpretBlock(func1.getBody().front());
  Semantics::State s2f = s2.interpretBlock(func2.getBody().front());

  auto t1 = std::chrono::high_resolution_clock::now();
  llvm::outs() << "Interpretation: "
               << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  //--------------------------------------------------------------------------
  // Emit AbstractFp axioms and check equivalence.
  //--------------------------------------------------------------------------
  s1f.fpReg->addAxioms(solver);
  s2f.fpReg->addAxioms(solver);

  // Assert that at least one output memory differs between the two programs,
  // compared POINTWISE at a single fresh symbolic witness address. store() now
  // builds each heap as a z3::lambda, so we must NOT use `array != array`
  // (array extensionality over lambdas returns `unknown`); select(lambda,
  // witness) β-reduces to a quantifier-free byte formula. (src and tgt come
  // from different functions, so we keep the explicit srcArgs↔tgtArgs pairing
  // rather than calling Semantics::checkEquivalence, which assumes shared args.)
  z3::expr witness = ctx.bv_const("__witness_addr", 64);
  z3::expr anyDiffers = ctx.bool_val(false);
  for (unsigned i = 0; i < srcArgs.size(); ++i) {
    mlir::Value sa = srcArgs[i];
    mlir::Value ta = tgtArgs[i];
    auto srcMemIt = s1f.ptrArgToMem.find(sa);
    auto tgtMemIt = s2f.ptrArgToMem.find(ta);
    if (srcMemIt != s1f.ptrArgToMem.end() &&
        tgtMemIt != s2f.ptrArgToMem.end()) {
      anyDiffers = anyDiffers ||
                   (z3::select(s1f.memState.mems.at(srcMemIt->second).array,
                               witness) !=
                    z3::select(s2f.memState.mems.at(tgtMemIt->second).array,
                               witness));
    }
  }
  solver.add(anyDiffers);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto result = solver.check();
  auto t3 = std::chrono::high_resolution_clock::now();
  llvm::outs() << "Solver: "
               << std::chrono::duration<double>(t3 - t2).count() << " s\n";

  if (result == z3::unsat) {
    llvm::outs() << "EQUIVALENT\n";
    return 0;
  } else if (result == z3::sat) {
    llvm::outs() << "NOT EQUIVALENT — counterexample found\n";
    llvm::outs() << solver.get_model().to_string().c_str() << "\n";
    return 1;
  } else {
    llvm::outs() << "UNKNOWN (solver timed out or gave up)\n";
    return 2;
  }
}
