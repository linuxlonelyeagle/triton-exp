#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<32> : tensor<32x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c32_i32 : i32
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %4 = tt.splat %2 : i32 -> tensor<32xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<32xi32, #blocked>
    %6 = arith.muli %1, %c32_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<32xi32, #blocked>
    %8 = arith.addi %7, %3 : tensor<32xi32, #blocked>
    %9 = triton_gpu.convert_layout %5 : tensor<32xi32, #blocked> -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %11 = triton_gpu.convert_layout %10 : tensor<32x1xi32, #blocked2> -> tensor<32x1xi32, #blocked3>
    %12 = tt.splat %arg9 : i32 -> tensor<32x1xi32, #blocked3>
    %13 = arith.muli %11, %12 : tensor<32x1xi32, #blocked3>
    %14 = triton_gpu.convert_layout %3 : tensor<32xi32, #blocked> -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x32xi32, #blocked4>
    %16 = triton_gpu.convert_layout %15 : tensor<1x32xi32, #blocked4> -> tensor<1x32xi32, #blocked1>
    %17 = tt.broadcast %13 : tensor<32x1xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
    %18 = triton_gpu.convert_layout %17 : tensor<32x32xi32, #blocked3> -> tensor<32x32xi32, #blocked1>
    %19 = tt.broadcast %16 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %20 = arith.addi %18, %19 : tensor<32x32xi32, #blocked1>
    %21 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %22 = tt.addptr %21, %20 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %23 = triton_gpu.convert_layout %3 : tensor<32xi32, #blocked> -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %25 = triton_gpu.convert_layout %24 : tensor<32x1xi32, #blocked2> -> tensor<32x1xi32, #blocked3>
    %26 = tt.splat %arg10 : i32 -> tensor<32x1xi32, #blocked3>
    %27 = arith.muli %25, %26 : tensor<32x1xi32, #blocked3>
    %28 = triton_gpu.convert_layout %8 : tensor<32xi32, #blocked> -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %29 = tt.expand_dims %28 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x32xi32, #blocked4>
    %30 = triton_gpu.convert_layout %29 : tensor<1x32xi32, #blocked4> -> tensor<1x32xi32, #blocked1>
    %31 = tt.broadcast %27 : tensor<32x1xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
    %32 = triton_gpu.convert_layout %31 : tensor<32x32xi32, #blocked3> -> tensor<32x32xi32, #blocked1>
    %33 = tt.broadcast %30 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %34 = arith.addi %32, %33 : tensor<32x32xi32, #blocked1>
    %35 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %36 = tt.addptr %35, %34 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %37 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>, #blocked>
    %38 = tt.addptr %37, %8 : tensor<32x!tt.ptr<f16>, #blocked>, tensor<32xi32, #blocked>
    %39 = arith.addi %arg8, %c31_i32 : i32
    %40 = arith.divsi %39, %c32_i32 : i32
    %41 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked3>
    %42 = arith.cmpi slt, %11, %41 : tensor<32x1xi32, #blocked3>
    %43 = tt.broadcast %42 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
    %44 = triton_gpu.convert_layout %43 : tensor<32x32xi1, #blocked3> -> tensor<32x32xi1, #blocked1>
    %45 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked1>
    %46 = arith.cmpi slt, %30, %45 : tensor<1x32xi32, #blocked1>
    %47 = tt.broadcast %46 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %48 = arith.muli %arg10, %c32_i32 : i32
    %49 = tt.splat %48 : i32 -> tensor<32x32xi32, #blocked1>
    %50:3 = scf.for %arg12 = %c0_i32 to %40 step %c1_i32 iter_args(%arg13 = %cst_2, %arg14 = %22, %arg15 = %36) -> (tensor<32x32xf32, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %82 = arith.muli %arg12, %c32_i32 : i32
      %83 = arith.subi %arg8, %82 : i32
      %84 = tt.splat %83 : i32 -> tensor<1x32xi32, #blocked1>
      %85 = arith.cmpi slt, %16, %84 : tensor<1x32xi32, #blocked1>
      %86 = tt.broadcast %85 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %87 = arith.andi %44, %86 : tensor<32x32xi1, #blocked1>
      %88 = tt.load %arg14, %87, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      %89 = tt.splat %83 : i32 -> tensor<32x1xi32, #blocked3>
      %90 = arith.cmpi slt, %25, %89 : tensor<32x1xi32, #blocked3>
      %91 = tt.broadcast %90 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
      %92 = triton_gpu.convert_layout %91 : tensor<32x32xi1, #blocked3> -> tensor<32x32xi1, #blocked1>
      %93 = arith.andi %92, %47 : tensor<32x32xi1, #blocked1>
      %94 = tt.load %arg15, %93, %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      %95 = triton_gpu.convert_layout %88 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>>
      %96 = triton_gpu.convert_layout %94 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked5}>>
      %97 = triton_gpu.convert_layout %arg13 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #blocked5>
      %98 = tt.dot %95, %96, %97 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<32x32xf32, #blocked5>
      %99 = triton_gpu.convert_layout %98 : tensor<32x32xf32, #blocked5> -> tensor<32x32xf32, #blocked1>
      %100 = tt.addptr %arg14, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      %101 = tt.addptr %arg15, %49 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %99, %100, %101 : tensor<32x32xf32, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    %51 = tt.splat %arg7 : i32 -> tensor<32xi32, #blocked>
    %52 = arith.cmpi slt, %8, %51 : tensor<32xi32, #blocked>
    %53 = tt.load %38, %52, %cst : tensor<32x!tt.ptr<f16>, #blocked>
    %54 = arith.sitofp %arg4 : i32 to f32
    %55 = tt.splat %54 : f32 -> tensor<32x32xf32, #blocked1>
    %56 = arith.mulf %50#0, %55 : tensor<32x32xf32, #blocked1>
    %57 = arith.sitofp %arg5 : i32 to f16
    %58 = tt.splat %57 : f16 -> tensor<32xf16, #blocked>
    %59 = arith.mulf %53, %58 : tensor<32xf16, #blocked>
    %60 = triton_gpu.convert_layout %59 : tensor<32xf16, #blocked> -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %61 = tt.expand_dims %60 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x32xf16, #blocked4>
    %62 = triton_gpu.convert_layout %61 : tensor<1x32xf16, #blocked4> -> tensor<1x32xf16, #blocked1>
    %63 = arith.extf %62 : tensor<1x32xf16, #blocked1> to tensor<1x32xf32, #blocked1>
    %64 = tt.broadcast %63 : tensor<1x32xf32, #blocked1> -> tensor<32x32xf32, #blocked1>
    %65 = arith.addf %56, %64 : tensor<32x32xf32, #blocked1>
    %66 = arith.truncf %65 : tensor<32x32xf32, #blocked1> to tensor<32x32xf16, #blocked1>
    %67 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked3>
    %68 = arith.muli %67, %11 : tensor<32x1xi32, #blocked3>
    %69 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked3>
    %70 = tt.addptr %69, %68 : tensor<32x1x!tt.ptr<f16>, #blocked3>, tensor<32x1xi32, #blocked3>
    %71 = tt.broadcast %70 : tensor<32x1x!tt.ptr<f16>, #blocked3> -> tensor<32x32x!tt.ptr<f16>, #blocked3>
    %72 = triton_gpu.convert_layout %71 : tensor<32x32x!tt.ptr<f16>, #blocked3> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %73 = tt.addptr %72, %33 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %74 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked3>
    %75 = arith.cmpi slt, %11, %74 : tensor<32x1xi32, #blocked3>
    %76 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked1>
    %77 = arith.cmpi slt, %30, %76 : tensor<1x32xi32, #blocked1>
    %78 = tt.broadcast %75 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
    %79 = triton_gpu.convert_layout %78 : tensor<32x32xi1, #blocked3> -> tensor<32x32xi1, #blocked1>
    %80 = tt.broadcast %77 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %81 = arith.andi %79, %80 : tensor<32x32xi1, #blocked1>
    tt.store %73, %66, %81 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

