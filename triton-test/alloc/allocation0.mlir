// -test-print-allocation
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
    %0 = tt.broadcast %arg3 : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #blocked0>
    %1 = tt.broadcast %arg4 : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %cst = arith.constant dense<true> : tensor<128x32xi1, #blocked0>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked0>
    %cst_1 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_4 = arith.constant dense<4> : tensor<128x32xi32, #blocked0>
    %cst_5 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %2:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %0, %arg7 = %1, %arg8 = %cst_3) -> (tensor<128x32x!tt.ptr<f16>, #blocked0>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %3 = tt.load %arg6, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked0>
      %4 = triton_gpu.convert_layout %3 : (tensor<128x32xf16, #blocked0>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %5 = tt.load %arg7, %cst_1, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %6 = triton_gpu.convert_layout %5 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %7 = tt.dot %4, %6, %arg8 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x128xf32, #mma>
      %8 = tt.addptr %arg6, %cst_4 : tensor<128x32x!tt.ptr<f16>, #blocked0>, tensor<128x32xi32, #blocked0>
      %9 = tt.addptr %arg7, %cst_5 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %8, %9, %7 : tensor<128x32x!tt.ptr<f16>, #blocked0>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    return
  }
}