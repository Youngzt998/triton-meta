#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @m(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dotA>
    %b = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dotB>
    %c = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %d0 = tt.dot %a, %b, %c : tensor<64x64xf16, #dotA> * tensor<64x64xf16, #dotB> -> tensor<64x64xf32, #blocked>
    %d1 = tt.dot %a, %b, %d0 : tensor<64x64xf16, #dotA> * tensor<64x64xf16, #dotB> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}
