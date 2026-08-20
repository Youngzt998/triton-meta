// Same defect as scan-cta-min.mlir, but the scan axis fits in one warp
// (warpsPerCTA[axis] == 1), so emitFastScan takes AddPartialReduceOneWarp and
// the identical assertion fires at ScanOpToLLVM.cpp:265 instead of :159.
// Here the CTA split is on the NON-scan axis, which is enough on its own:
// getNonAxisNumBlocks() also divides the whole shape by the per-CTA tile.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_cta_min_onewarp(%arg0: !tt.ptr<i32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
    %3 = "tt.scan"(%2) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %4 = arith.addi %a, %b : i32
      tt.scan.return %4 : i32
    }) : (tensor<256x128xi32, #blocked>) -> tensor<256x128xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x128x!tt.ptr<i32>, #blocked>
    tt.store %5, %3 : tensor<256x128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
