#ifndef TILE_SMT_MEMORY_H
#define TILE_SMT_MEMORY_H

// tile-smt core — byte-addressable memory + masked windowed load/store.
// MLIR-free: this header depends only on Z3 and the tile-smt neutral types.
// The Z3 encoding (store lambda + readBytes byte packing) is preserved exactly
// from the pre-M0 Triton-coupled implementation.

#include "semantics/Types.h"
#include "semantics/Value.h"

#include <map>
#include <string>
#include <z3++.h>

namespace tile_smt {

// Byte-addressable heap: Array(BitVec(64), BitVec(8)).
//
// load() returns a new Tensor; store() mutates array in place and returns *this.
//
// FP element handling: under FPMode::Abstract, FP values are BitVec(width) ids
// (see AbstractFp.h) and are read/written as raw bytes like integers. The
// abstract semantics is recovered when an operation handler applies AbstractFp's
// uninterpreted functions to a loaded value.
//
// The Real / IntegerRange / FPA modes are not yet wired in to load/store.
//
// Memory holds z3::context by reference: it is copy-CONSTRUCTIBLE (tests rely on
// `Memory m1 = init;`) but NOT copy-assignable. Insert into MemState via
// emplace / piecewise_construct, never by assignment.
class Memory {
public:
  z3::context &ctx;
  FPMode       fpMode;
  z3::expr     array; // Array(BitVec(64), BitVec(8))

  // Creates a fresh unconstrained symbolic memory named `name`.
  // Use distinct names for the two programs being compared.
  Memory(z3::context &ctx, FPMode fpMode, const std::string &name = "mem");

  // Load a tile from memory under a mask.
  //   ptrTile  : tile of byte addresses,  Array(BitVec(32), BitVec(64))
  //   maskTile : boolean mask,             Array(BitVec(32), Bool)
  //   otherTile: passthrough when !mask[i]; defines result shape and elem
  // Returns a Tensor with the same shape and elem as otherTile.
  Tensor load(const Tensor &ptrTile,
              const Tensor &maskTile,
              const Tensor &otherTile) const;

  // Store a tile to memory under a mask (mutates array in place).
  //   ptrTile  : tile of byte addresses, Array(BitVec(32), BitVec(64))
  //   valTile  : values to write
  //   maskTile : boolean mask; only writes where mask[i]=true
  Memory &store(const Tensor &ptrTile,
                const Tensor &valTile,
                const Tensor &maskTile);

private:
  // Read byteWidth bytes from array at addr (little-endian).
  // Returns BitVec(byteWidth * 8).
  z3::expr readBytes(z3::expr addr, unsigned byteWidth) const;
};

// One independent symbolic Memory per pointer memory / address space, keyed by
// the opaque MemId a builder assigns. Replaces the old State::ptrMems (which
// keyed per-arg memories by the source pointer-argument SSA value).
//
// MemId is a plain enum, so it is a std::map key directly (no comparator).
// Memory is not assignable, so entries are inserted via emplace / piecewise.
struct MemState {
  std::map<MemId, Memory> mems;
};

} // namespace tile_smt

#endif // TILE_SMT_MEMORY_H
