#include "../../bin/RegisterTritonDialects.h"

#include "builder/mlir/State.h"
#include "semantics/Equivalence.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <utility>
#include <vector>
#include <z3++.h>

static llvm::cl::opt<std::string>
    inputFile1(llvm::cl::Positional, llvm::cl::desc("<first input mlir file>"),
               llvm::cl::Required);

static llvm::cl::opt<std::string>
    inputFile2(llvm::cl::Positional, llvm::cl::desc("<second input mlir file>"),
               llvm::cl::Required);

/// Parse an MLIR file and return the module.
static mlir::OwningOpRef<mlir::ModuleOp>
parseMLIRFile(const std::string &filename, mlir::MLIRContext &context) {
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

void floatingSATTest() {
  z3::context ctx;
  z3::sort fp32 = ctx.fpa_sort(
      8, 24); // IEEE 754 float: 8 exp bits, 24 sig bits (including hidden bit)
  z3::expr a = ctx.constant("a", fp32);
  z3::expr b = ctx.constant("b", fp32);
  z3::expr rm = ctx.fpa_rounding_mode(); // RNE (round nearest, ties to even)

  z3::expr lhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, a, b)); // a + b
  z3::expr rhs = z3::expr(ctx, Z3_mk_fpa_add(ctx, rm, b, a)); // b + a

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

    z3::expr mulLhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, a2, b2)); // a * b
    z3::expr mulRhs = z3::expr(ctx2, Z3_mk_fpa_mul(ctx2, rm2, b2, a2)); // b * a

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

void realSATTest() {
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

  mlir::OwningOpRef<mlir::ModuleOp> module1 =
      parseMLIRFile(inputFile1, context);
  if (!module1) {
    llvm::errs() << "Failed to parse: " << inputFile1 << "\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module2 =
      parseMLIRFile(inputFile2, context);
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

  llvm::outs() << "Validating: " << func1.getName() << " vs " << func2.getName()
               << "\n";

  //--------------------------------------------------------------------------
  // Build symbolic initial states.
  //--------------------------------------------------------------------------
  z3::context ctx;
  z3::solver solver(ctx);

  auto srcArgs = func1.getBody().getArguments();
  auto tgtArgs = func2.getBody().getArguments();

  if (srcArgs.size() != tgtArgs.size()) {
    llvm::errs() << "Argument count mismatch.\n";
    return 1;
  }

  // One core Context per program (src, tgt), both sharing the single
  // z3::context so program-ID and FP function symbols are shared across the two
  // programs — exactly the pre-M0 behavior (which used two AbstractFpRegistry
  // objects on one z3::context).
  tile_smt::Context srcCtx(ctx, Semantics::FPMode::Abstract);
  tile_smt::Context tgtCtx(ctx, Semantics::FPMode::Abstract);

  Semantics::State s1 = Semantics::State::initFromFunc(srcArgs, srcCtx, "src");
  Semantics::State s2 = Semantics::State::initFromFunc(tgtArgs, tgtCtx, "tgt");

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
      std::visit(
          [&](auto &v1) {
            using T = std::decay_t<decltype(v1)>;
            solver.add(v1.e == std::get<T>(s2.env.lookup(ta)).e);
          },
          s1.env.lookup(sa));
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
  s1f.context.fp().addAxioms(solver);
  s2f.context.fp().addAxioms(solver);

  // Pair the two programs' output memories positionally by kernel argument:
  // src arg i's memory ↔ tgt arg i's memory. src and tgt come from different
  // functions, so the adapter builds this MemId pairing explicitly and hands it
  // to the MLIR-free core (which then checks that at least one paired memory
  // differs at a fresh symbolic witness address; see tile_smt::checkEquivalence
  // for why we compare pointwise instead of `array != array`).
  std::vector<std::pair<tile_smt::MemId, tile_smt::MemId>> pairing;
  for (unsigned i = 0; i < srcArgs.size(); ++i) {
    auto srcMemIt = s1f.ptrArgToMem.find(srcArgs[i]);
    auto tgtMemIt = s2f.ptrArgToMem.find(tgtArgs[i]);
    if (srcMemIt != s1f.ptrArgToMem.end() &&
        tgtMemIt != s2f.ptrArgToMem.end()) {
      pairing.emplace_back(srcMemIt->second, tgtMemIt->second);
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto result =
      tile_smt::checkEquivalence(s1f.memState, s2f.memState, pairing, solver);
  auto t3 = std::chrono::high_resolution_clock::now();
  llvm::outs() << "Solver: " << std::chrono::duration<double>(t3 - t2).count()
               << " s\n";

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
