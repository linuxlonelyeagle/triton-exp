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
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %18 = tt.broadcast %15 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %19 = tt.broadcast %17 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %20 = arith.addi %18, %19 : tensor<32x32xi32, #blocked>
    %21 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %22 = tt.addptr %21, %20 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %23 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %25 = tt.splat %arg10 : i32 -> tensor<32x1xi32, #blocked>
    %26 = arith.muli %24, %25 : tensor<32x1xi32, #blocked>
    %27 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %28 = tt.broadcast %26 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %29 = tt.broadcast %27 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %30 = arith.addi %28, %29 : tensor<32x32xi32, #blocked>
    %31 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %32 = tt.addptr %31, %30 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %33 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %34 = tt.addptr %33, %12 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %35 = arith.addi %arg8, %c31_i32 : i32
    %36 = arith.divsi %35, %c32_i32 : i32
    %37 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked>
    %38 = arith.cmpi slt, %13, %37 : tensor<32x1xi32, #blocked>
    %39 = tt.broadcast %38 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %40 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked>
    %41 = arith.cmpi slt, %27, %40 : tensor<1x32xi32, #blocked>
    %42 = tt.broadcast %41 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %43 = arith.muli %arg10, %c32_i32 : i32
    %44 = tt.splat %43 : i32 -> tensor<32x32xi32, #blocked>
    %45:3 = scf.for %arg12 = %c0_i32 to %36 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %22, %arg15 = %32) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>)  : i32 {
      %74 = arith.muli %arg12, %c32_i32 : i32
      %75 = arith.subi %arg8, %74 : i32
      %76 = tt.splat %75 : i32 -> tensor<1x32xi32, #blocked>
      %77 = arith.cmpi slt, %17, %76 : tensor<1x32xi32, #blocked>
      %78 = tt.broadcast %77 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %79 = arith.andi %39, %78 : tensor<32x32xi1, #blocked>
      %80 = tt.load %arg14, %79, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %81 = tt.splat %75 : i32 -> tensor<32x1xi32, #blocked>
      %82 = arith.cmpi slt, %24, %81 : tensor<32x1xi32, #blocked>
      %83 = tt.broadcast %82 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %84 = arith.andi %83, %42 : tensor<32x32xi1, #blocked>
      %85 = tt.load %arg15, %84, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %86 = triton_gpu.convert_layout %80 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %87 = triton_gpu.convert_layout %85 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %88 = tt.dot %86, %87, %arg13 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %89 = tt.addptr %arg14, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %90 = tt.addptr %arg15, %44 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %88, %89, %90 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>
    }
    %46 = tt.splat %arg7 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %47 = arith.cmpi slt, %12, %46 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %48 = tt.load %34, %47, %cst_2 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %49 = arith.sitofp %arg4 : i32 to f32
    %50 = tt.splat %49 : f32 -> tensor<32x32xf32, #mma>
    %51 = arith.mulf %45#0, %50 : tensor<32x32xf32, #mma>
    %52 = arith.sitofp %arg5 : i32 to f16
    %53 = tt.splat %52 : f16 -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %54 = arith.mulf %48, %53 : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %55 = tt.expand_dims %54 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf16, #mma>
    %56 = arith.extf %55 : tensor<1x32xf16, #mma> to tensor<1x32xf32, #mma>
    %57 = tt.broadcast %56 : tensor<1x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %58 = arith.addf %51, %57 : tensor<32x32xf32, #mma>
    %59 = arith.truncf %58 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %60 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked>
    %61 = arith.muli %60, %13 : tensor<32x1xi32, #blocked>
    %62 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %63 = tt.addptr %62, %61 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %64 = tt.broadcast %63 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %65 = tt.addptr %64, %29 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %66 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked>
    %67 = arith.cmpi slt, %13, %66 : tensor<32x1xi32, #blocked>
    %68 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked>
    %69 = arith.cmpi slt, %27, %68 : tensor<1x32xi32, #blocked>
    %70 = tt.broadcast %67 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %71 = tt.broadcast %69 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %72 = arith.andi %70, %71 : tensor<32x32xi1, #blocked>
    %73 = triton_gpu.convert_layout %59 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked>
    tt.store %65, %73, %72 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

