#shared = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module {
  func @preallocate(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #shared>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #shared>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #shared>
    %0 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #shared>, tensor<16x16xf16, #shared>) -> tensor<32x16xf16, #shared>
    %1 = tt.cat %cst, %cst_1 {axis = 0 : i64} : (tensor<16x16xf16, #shared>, tensor<16x16xf16, #shared>) -> tensor<32x16xf16, #shared>
    %2 = tt.cat %cst_0, %cst_1 {axis = 0 : i64} : (tensor<16x16xf16, #shared>, tensor<16x16xf16, #shared>) -> tensor<32x16xf16, #shared>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #shared>
    %3 = tt.cat %0, %cst_2 {axis = 0 : i64} : (tensor<32x16xf16, #shared>, tensor<32x16xf16, #shared>) -> tensor<64x16xf16, #shared>
    %4 = tt.cat %1, %cst_2 {axis = 0 : i64} : (tensor<32x16xf16, #shared>, tensor<32x16xf16, #shared>) -> tensor<64x16xf16, #shared>
    %5 = tt.cat %2, %cst_2 {axis = 0 : i64} : (tensor<32x16xf16, #shared>, tensor<32x16xf16, #shared>) -> tensor<64x16xf16, #shared>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #shared>
    %6 = tt.cat %3, %cst_3 {axis = 0 : i64} : (tensor<64x16xf16, #shared>, tensor<64x16xf16, #shared>) -> tensor<128x16xf16, #shared>
    %7 = tt.cat %4, %cst_3 {axis = 0 : i64} : (tensor<64x16xf16, #shared>, tensor<64x16xf16, #shared>) -> tensor<128x16xf16, #shared>
    %8 = tt.cat %5, %cst_3 {axis = 0 : i64} : (tensor<64x16xf16, #shared>, tensor<64x16xf16, #shared>) -> tensor<128x16xf16, #shared>
    return
  }
}
