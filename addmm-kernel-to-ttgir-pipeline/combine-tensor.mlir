#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_1 = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c32_i32 : i32
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %6 = tt.splat %2 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = arith.addi %6, %3 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %8 = arith.muli %1, %c32_i32 : i32
    %9 = tt.splat %8 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.splat %8 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %11 = arith.addi %9, %4 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %12 = arith.addi %10, %5 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %13 = tt.expand_dims %7 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %14 = tt.splat %arg9 : i32 -> tensor<32x1xi32, #blocked>
    %15 = arith.muli %13, %14 : tensor<32x1xi32, #blocked>
    %16 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %17 = tt.broadcast %15 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %18 = tt.broadcast %16 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %19 = arith.addi %17, %18 : tensor<32x32xi32, #blocked>
    %20 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %21 = tt.addptr %20, %19 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %22 = tt.expand_dims %3 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %23 = tt.splat %arg10 : i32 -> tensor<32x1xi32, #blocked>
    %24 = arith.muli %22, %23 : tensor<32x1xi32, #blocked>
    %25 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %26 = tt.broadcast %24 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %27 = tt.broadcast %25 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %28 = arith.addi %26, %27 : tensor<32x32xi32, #blocked>
    %29 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %30 = tt.addptr %29, %28 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %31 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %32 = tt.addptr %31, %12 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %33 = arith.addi %arg8, %c31_i32 : i32
    %34 = arith.divsi %33, %c32_i32 : i32
    %35 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked>
    %36 = arith.cmpi slt, %13, %35 : tensor<32x1xi32, #blocked>
    %37 = tt.broadcast %36 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %38 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked>
    %39 = arith.cmpi slt, %25, %38 : tensor<1x32xi32, #blocked>
    %40 = tt.broadcast %39 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %41 = arith.muli %arg10, %c32_i32 : i32
    %42 = tt.splat %41 : i32 -> tensor<32x32xi32, #blocked>
    %43:3 = scf.for %arg12 = %c0_i32 to %34 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %21, %arg15 = %30) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>)  : i32 {
      %66 = arith.muli %arg12, %c32_i32 : i32
      %67 = arith.subi %arg8, %66 : i32
      %68 = tt.splat %67 : i32 -> tensor<1x32xi32, #blocked>
      %69 = arith.cmpi slt, %16, %68 : tensor<1x32xi32, #blocked>
      %70 = tt.broadcast %69 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %71 = arith.andi %37, %70 : tensor<32x32xi1, #blocked>
      %72 = tt.load %arg14, %71, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %73 = tt.splat %67 : i32 -> tensor<32x1xi32, #blocked>
      %74 = arith.cmpi slt, %22, %73 : tensor<32x1xi32, #blocked>
      %75 = tt.broadcast %74 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %76 = arith.andi %75, %40 : tensor<32x32xi1, #blocked>
      %77 = tt.load %arg15, %76, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %78 = triton_gpu.convert_layout %72 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %79 = triton_gpu.convert_layout %77 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %80 = tt.dot %78, %79, %arg13 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %81 = tt.addptr %arg14, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %82 = tt.addptr %arg15, %42 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %80, %81, %82 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>
    }
    %44 = tt.splat %arg7 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %45 = arith.cmpi slt, %12, %44 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %46 = tt.load %32, %45, %cst_2 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %47 = arith.sitofp %arg4 : i32 to f32
    %48 = tt.splat %47 : f32 -> tensor<32x32xf32, #mma>
    %49 = arith.mulf %43#0, %48 : tensor<32x32xf32, #mma>
    %50 = arith.sitofp %arg5 : i32 to f16
    %51 = tt.splat %50 : f16 -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %52 = arith.mulf %46, %51 : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %53 = tt.expand_dims %52 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf16, #mma>
    %54 = arith.extf %53 : tensor<1x32xf16, #mma> to tensor<1x32xf32, #mma>
    %55 = tt.broadcast %54 : tensor<1x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %56 = arith.addf %49, %55 : tensor<32x32xf32, #mma>
    %57 = arith.truncf %56 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %58 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked>
    %59 = arith.muli %58, %13 : tensor<32x1xi32, #blocked>
    %60 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %61 = tt.addptr %60, %59 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %62 = tt.broadcast %61 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %63 = tt.addptr %62, %27 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %64 = arith.andi %37, %40 : tensor<32x32xi1, #blocked>
    %65 = triton_gpu.convert_layout %57 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked>
    tt.store %63, %65, %64 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

