// tt.scan on a 1-D tensor whose axis is split across the CTAs of a cluster.
// num_ctas = 2, so the 256-element axis is 128 elements per CTA -> 1 value per
// thread.  ScanLoweringHelper::getAxisNumBlocks() divides the WHOLE shape (256)
// by the PER-CTA capacity (128) and gets 2, so the lowering expects 2 values
// per thread and the assertion in AddPartialReduce* fails.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_cta_min(%arg0: !tt.ptr<i32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %1 = "tt.scan"(%0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %2 = arith.addi %a, %b : i32
      tt.scan.return %2 : i32
    }) : (tensor<256xi32, #blocked>) -> tensor<256xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
    tt.store %3, %1 : tensor<256x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
