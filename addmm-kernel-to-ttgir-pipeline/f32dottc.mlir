#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
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
      %88 = arith.muli %arg12, %c32_i32 : i32
      %89 = arith.subi %arg8, %88 : i32
      %90 = tt.splat %89 : i32 -> tensor<1x32xi32, #blocked1>
      %91 = arith.cmpi slt, %16, %90 : tensor<1x32xi32, #blocked1>
      %92 = tt.broadcast %91 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %93 = arith.andi %44, %92 : tensor<32x32xi1, #blocked1>
      %94 = triton_gpu.convert_layout %arg14 : tensor<32x32x!tt.ptr<f16>, #blocked1> -> tensor<32x32x!tt.ptr<f16>, #blocked5>
      %95 = triton_gpu.convert_layout %93 : tensor<32x32xi1, #blocked1> -> tensor<32x32xi1, #blocked5>
      %96 = triton_gpu.convert_layout %cst_0 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked5>
      %97 = tt.load %94, %95, %96 : tensor<32x32x!tt.ptr<f16>, #blocked5>
      %98 = triton_gpu.convert_layout %97 : tensor<32x32xf16, #blocked5> -> tensor<32x32xf16, #blocked1>
      %99 = tt.splat %89 : i32 -> tensor<32x1xi32, #blocked3>
      %100 = arith.cmpi slt, %25, %99 : tensor<32x1xi32, #blocked3>
      %101 = tt.broadcast %100 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
      %102 = triton_gpu.convert_layout %101 : tensor<32x32xi1, #blocked3> -> tensor<32x32xi1, #blocked1>
      %103 = arith.andi %102, %47 : tensor<32x32xi1, #blocked1>
      %104 = triton_gpu.convert_layout %arg15 : tensor<32x32x!tt.ptr<f16>, #blocked1> -> tensor<32x32x!tt.ptr<f16>, #blocked5>
      %105 = triton_gpu.convert_layout %103 : tensor<32x32xi1, #blocked1> -> tensor<32x32xi1, #blocked5>
      %106 = triton_gpu.convert_layout %cst_0 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked5>
      %107 = tt.load %104, %105, %106 : tensor<32x32x!tt.ptr<f16>, #blocked5>
      %108 = triton_gpu.convert_layout %107 : tensor<32x32xf16, #blocked5> -> tensor<32x32xf16, #blocked1>
      %109 = triton_gpu.convert_layout %98 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>>
      %110 = triton_gpu.convert_layout %108 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>>
      %111 = triton_gpu.convert_layout %arg13 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #blocked6>
      %112 = tt.dot %109, %110, %111 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<32x32xf32, #blocked6>
      %113 = triton_gpu.convert_layout %112 : tensor<32x32xf32, #blocked6> -> tensor<32x32xf32, #blocked1>
      %114 = tt.addptr %arg14, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      %115 = tt.addptr %arg15, %49 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %113, %114, %115 : tensor<32x32xf32, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    %51 = tt.splat %arg7 : i32 -> tensor<32xi32, #blocked>
    %52 = arith.cmpi slt, %8, %51 : tensor<32xi32, #blocked>
    %53 = triton_gpu.convert_layout %38 : tensor<32x!tt.ptr<f16>, #blocked> -> tensor<32x!tt.ptr<f16>, #blocked>
    %54 = triton_gpu.convert_layout %52 : tensor<32xi1, #blocked> -> tensor<32xi1, #blocked>
    %55 = triton_gpu.convert_layout %cst : tensor<32xf16, #blocked> -> tensor<32xf16, #blocked>
    %56 = tt.load %53, %54, %55 : tensor<32x!tt.ptr<f16>, #blocked>
    %57 = arith.sitofp %arg4 : i32 to f32
    %58 = tt.splat %57 : f32 -> tensor<32x32xf32, #blocked1>
    %59 = arith.mulf %50#0, %58 : tensor<32x32xf32, #blocked1>
    %60 = arith.sitofp %arg5 : i32 to f16
    %61 = tt.splat %60 : f16 -> tensor<32xf16, #blocked>
    %62 = arith.mulf %56, %61 : tensor<32xf16, #blocked>
    %63 = triton_gpu.convert_layout %62 : tensor<32xf16, #blocked> -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %64 = tt.expand_dims %63 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x32xf16, #blocked4>
    %65 = triton_gpu.convert_layout %64 : tensor<1x32xf16, #blocked4> -> tensor<1x32xf16, #blocked1>
    %66 = arith.extf %65 : tensor<1x32xf16, #blocked1> to tensor<1x32xf32, #blocked1>
    %67 = tt.broadcast %66 : tensor<1x32xf32, #blocked1> -> tensor<32x32xf32, #blocked1>
    %68 = arith.addf %59, %67 : tensor<32x32xf32, #blocked1>
    %69 = arith.truncf %68 : tensor<32x32xf32, #blocked1> to tensor<32x32xf16, #blocked1>
    %70 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked3>
    %71 = arith.muli %70, %11 : tensor<32x1xi32, #blocked3>
    %72 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked3>
    %73 = tt.addptr %72, %71 : tensor<32x1x!tt.ptr<f16>, #blocked3>, tensor<32x1xi32, #blocked3>
    %74 = tt.broadcast %73 : tensor<32x1x!tt.ptr<f16>, #blocked3> -> tensor<32x32x!tt.ptr<f16>, #blocked3>
    %75 = triton_gpu.convert_layout %74 : tensor<32x32x!tt.ptr<f16>, #blocked3> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %76 = tt.addptr %75, %33 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %77 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked3>
    %78 = arith.cmpi slt, %11, %77 : tensor<32x1xi32, #blocked3>
    %79 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #blocked1>
    %80 = arith.cmpi slt, %30, %79 : tensor<1x32xi32, #blocked1>
    %81 = tt.broadcast %78 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
    %82 = triton_gpu.convert_layout %81 : tensor<32x32xi1, #blocked3> -> tensor<32x32xi1, #blocked1>
    %83 = tt.broadcast %80 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %84 = arith.andi %82, %83 : tensor<32x32xi1, #blocked1>
    %85 = triton_gpu.convert_layout %76 : tensor<32x32x!tt.ptr<f16>, #blocked1> -> tensor<32x32x!tt.ptr<f16>, #blocked5>
    %86 = triton_gpu.convert_layout %69 : tensor<32x32xf16, #blocked1> -> tensor<32x32xf16, #blocked5>
    %87 = triton_gpu.convert_layout %84 : tensor<32x32xi1, #blocked1> -> tensor<32x32xi1, #blocked5>
    tt.store %85, %86, %87 : tensor<32x32x!tt.ptr<f16>, #blocked5>
    tt.return
  }
}

