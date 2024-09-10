#blocked = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked1>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %cst_2 = arith.constant dense<32> : tensor<32x32xi32, #blocked1>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c32_i32 : i32
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.splat %2 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %7 = arith.addi %6, %3 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %8 = arith.muli %1, %c32_i32 : i32
    %9 = tt.splat %8 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %10 = tt.splat %8 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %11 = arith.addi %9, %4 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %12 = arith.addi %10, %5 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.expand_dims %7 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %14 = tt.splat %arg9 : i32 -> tensor<32x1xi32, #blocked1>
    %15 = arith.muli %13, %14 : tensor<32x1xi32, #blocked1>
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %18 = tt.broadcast %15 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %19 = tt.broadcast %17 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %20 = arith.addi %18, %19 : tensor<32x32xi32, #blocked1>
    %21 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %22 = tt.addptr %21, %20 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %23 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %25 = tt.splat %arg10 : i32 -> tensor<32x1xi32, #blocked1>
    %26 = arith.muli %24, %25 : tensor<32x1xi32, #blocked1>
    %27 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %28 = tt.broadcast %26 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %30 = arith.addi %28, %29 : tensor<32x32xi32, #blocked1>
    %31 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %32 = tt.addptr %31, %30 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %33 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %34 = tt.addptr %33, %12 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %35 = arith.addi %arg8, %c31_i32 : i32
    %36 = arith.divsi %35, %c32_i32 : i32
    %37 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked1>
    %38 = arith.cmpi slt, %13, %37 : tensor<32x1xi32, #blocked1>
    %39 = tt.broadcast %38 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %40 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked1>
    %41 = arith.cmpi slt, %27, %40 : tensor<1x32xi32, #blocked1>
    %42 = tt.broadcast %41 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %43 = arith.muli %arg10, %c32_i32 : i32
    %44 = tt.splat %43 : i32 -> tensor<32x32xi32, #blocked1>
    %45:3 = scf.for %arg12 = %c0_i32 to %36 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %22, %arg15 = %32) -> (tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %74 = arith.muli %arg12, %c32_i32 : i32
      %75 = arith.subi %arg8, %74 : i32
      %76 = tt.splat %75 : i32 -> tensor<1x32xi32, #blocked1>
      %77 = arith.cmpi slt, %17, %76 : tensor<1x32xi32, #blocked1>
      %78 = tt.broadcast %77 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %79 = arith.andi %39, %78 : tensor<32x32xi1, #blocked1>
      %80 = tt.load %arg14, %79, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      %81 = tt.splat %75 : i32 -> tensor<32x1xi32, #blocked1>
      %82 = arith.cmpi slt, %24, %81 : tensor<32x1xi32, #blocked1>
      %83 = tt.broadcast %82 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %84 = arith.andi %83, %42 : tensor<32x32xi1, #blocked1>
      %85 = tt.load %arg15, %84, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      %86 = triton_gpu.convert_layout %80 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %87 = triton_gpu.convert_layout %85 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %88 = triton_gpu.convert_layout %arg13 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
      %89 = triton_gpu.convert_layout %86 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %90 = triton_gpu.convert_layout %87 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %91 = tt.dot %89, %90, %88 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %92 = triton_gpu.convert_layout %91 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      %93 = tt.addptr %arg14, %cst_2 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      %94 = tt.addptr %arg15, %44 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %92, %93, %94 : tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    %46 = tt.splat %arg7 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %47 = arith.cmpi slt, %12, %46 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %48 = tt.load %34, %47, %cst_1 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %49 = arith.sitofp %arg4 : i32 to f32
    %50 = tt.splat %49 : f32 -> tensor<32x32xf32, #blocked>
    %51 = arith.mulf %45#0, %50 : tensor<32x32xf32, #blocked>
    %52 = arith.sitofp %arg5 : i32 to f16
    %53 = tt.splat %52 : f16 -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %54 = arith.mulf %48, %53 : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %55 = tt.expand_dims %54 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf16, #blocked>
    %56 = arith.extf %55 : tensor<1x32xf16, #blocked> to tensor<1x32xf32, #blocked>
    %57 = tt.broadcast %56 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
    %58 = arith.addf %51, %57 : tensor<32x32xf32, #blocked>
    %59 = arith.truncf %58 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %60 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked1>
    %61 = arith.muli %60, %13 : tensor<32x1xi32, #blocked1>
    %62 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked1>
    %63 = tt.addptr %62, %61 : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi32, #blocked1>
    %64 = tt.broadcast %63 : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %65 = tt.addptr %64, %29 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %66 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked1>
    %67 = arith.cmpi slt, %13, %66 : tensor<32x1xi32, #blocked1>
    %68 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked1>
    %69 = arith.cmpi slt, %27, %68 : tensor<1x32xi32, #blocked1>
    %70 = tt.broadcast %67 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %71 = tt.broadcast %69 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %72 = arith.andi %70, %71 : tensor<32x32xi1, #blocked1>
    %73 = triton_gpu.convert_layout %59 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #blocked1>
    tt.store %65, %73, %72 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

