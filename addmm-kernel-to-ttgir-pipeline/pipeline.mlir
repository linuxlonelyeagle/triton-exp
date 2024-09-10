#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
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
    %43 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %44 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %45 = arith.cmpi sgt, %34, %c0_i32 : i32
    %46 = tt.splat %arg8 : i32 -> tensor<1x32xi32, #blocked>
    %47 = arith.cmpi slt, %16, %46 : tensor<1x32xi32, #blocked>
    %48 = tt.broadcast %47 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %49 = arith.andi %37, %48 : tensor<32x32xi1, #blocked>
    %50 = triton_gpu.memdesc_subview %43[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %51 = tt.splat %45 : i1 -> tensor<32x32xi1, #blocked>
    %52 = arith.andi %51, %49 : tensor<32x32xi1, #blocked>
    %53 = triton_gpu.async_copy_global_to_local %21, %50 mask %52 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %54 = triton_gpu.async_commit_group %53
    %55 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #blocked>
    %56 = arith.cmpi slt, %22, %55 : tensor<32x1xi32, #blocked>
    %57 = tt.broadcast %56 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %58 = arith.andi %57, %40 : tensor<32x32xi1, #blocked>
    %59 = triton_gpu.memdesc_subview %44[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %60 = tt.splat %45 : i1 -> tensor<32x32xi1, #blocked>
    %61 = arith.andi %60, %58 : tensor<32x32xi1, #blocked>
    %62 = triton_gpu.async_copy_global_to_local %30, %59 mask %61 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %63 = triton_gpu.async_commit_group %62
    %64 = arith.cmpi sgt, %34, %c1_i32 : i32
    %65 = tt.addptr %21, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %66 = tt.addptr %30, %42 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %67 = arith.subi %arg8, %c32_i32 : i32
    %68 = tt.splat %67 : i32 -> tensor<1x32xi32, #blocked>
    %69 = arith.cmpi slt, %16, %68 : tensor<1x32xi32, #blocked>
    %70 = tt.broadcast %69 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %71 = arith.andi %37, %70 : tensor<32x32xi1, #blocked>
    %72 = triton_gpu.memdesc_subview %43[%c1_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %73 = tt.splat %64 : i1 -> tensor<32x32xi1, #blocked>
    %74 = arith.andi %73, %71 : tensor<32x32xi1, #blocked>
    %75 = triton_gpu.async_copy_global_to_local %65, %72 mask %74 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %76 = triton_gpu.async_commit_group %75
    %77 = tt.splat %67 : i32 -> tensor<32x1xi32, #blocked>
    %78 = arith.cmpi slt, %22, %77 : tensor<32x1xi32, #blocked>
    %79 = tt.broadcast %78 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %80 = arith.andi %79, %40 : tensor<32x32xi1, #blocked>
    %81 = triton_gpu.memdesc_subview %44[%c1_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %82 = tt.splat %64 : i1 -> tensor<32x32xi1, #blocked>
    %83 = arith.andi %82, %80 : tensor<32x32xi1, #blocked>
    %84 = triton_gpu.async_copy_global_to_local %66, %81 mask %83 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %85 = triton_gpu.async_commit_group %84
    %86 = triton_gpu.memdesc_subview %43[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %87 = triton_gpu.async_wait %63 {num = 2 : i32}
    %88 = triton_gpu.memdesc_subview %44[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %89:11 = scf.for %arg12 = %c0_i32 to %34 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %65, %arg15 = %66, %arg16 = %c1_i32, %arg17 = %c0_i32, %arg18 = %86, %arg19 = %87, %arg20 = %88, %arg21 = %87, %arg22 = %76, %arg23 = %85) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>, i32, i32, !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.async.token, !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.async.token, !triton_gpu.async.token, !triton_gpu.async.token)  : i32 {
      %113 = arith.subi %34, %c2_i32 : i32
      %114 = arith.cmpi slt, %arg12, %113 : i32
      %115 = triton_gpu.local_load %arg18 token %arg19 : !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf16, #blocked>
      %116 = triton_gpu.local_load %arg20 token %arg21 : !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf16, #blocked>
      %117 = triton_gpu.convert_layout %115 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %118 = triton_gpu.convert_layout %116 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %119 = tt.dot %117, %118, %arg13 : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %120 = tt.addptr %arg14, %cst_1 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %121 = tt.addptr %arg15, %42 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %122 = arith.addi %arg16, %c1_i32 : i32
      %123 = arith.cmpi slt, %122, %c2_i32 : i32
      %124 = arith.select %123, %122, %c0_i32 : i32
      %125 = arith.addi %arg12, %c2_i32 : i32
      %126 = arith.muli %125, %c32_i32 : i32
      %127 = arith.subi %arg8, %126 : i32
      %128 = tt.splat %127 : i32 -> tensor<1x32xi32, #blocked>
      %129 = arith.cmpi slt, %16, %128 : tensor<1x32xi32, #blocked>
      %130 = tt.broadcast %129 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %131 = arith.andi %37, %130 : tensor<32x32xi1, #blocked>
      %132 = triton_gpu.memdesc_subview %43[%124, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %133 = tt.splat %114 : i1 -> tensor<32x32xi1, #blocked>
      %134 = arith.andi %133, %131 : tensor<32x32xi1, #blocked>
      %135 = triton_gpu.async_copy_global_to_local %120, %132 mask %134 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %136 = triton_gpu.async_commit_group %135
      %137 = tt.splat %127 : i32 -> tensor<32x1xi32, #blocked>
      %138 = arith.cmpi slt, %22, %137 : tensor<32x1xi32, #blocked>
      %139 = tt.broadcast %138 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %140 = arith.andi %139, %40 : tensor<32x32xi1, #blocked>
      %141 = triton_gpu.memdesc_subview %44[%124, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %142 = tt.splat %114 : i1 -> tensor<32x32xi1, #blocked>
      %143 = arith.andi %142, %140 : tensor<32x32xi1, #blocked>
      %144 = triton_gpu.async_copy_global_to_local %121, %141 mask %143 other %cst_0 : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %145 = triton_gpu.async_commit_group %144
      %146 = arith.addi %arg17, %c1_i32 : i32
      %147 = arith.cmpi slt, %146, %c2_i32 : i32
      %148 = arith.select %147, %146, %c0_i32 : i32
      %149 = triton_gpu.memdesc_subview %43[%148, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %150 = triton_gpu.async_wait %arg23 {num = 2 : i32}
      %151 = triton_gpu.memdesc_subview %44[%148, %c0_i32, %c0_i32] : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %119, %120, %121, %124, %148, %149, %150, %151, %150, %136, %145 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>, i32, i32, !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.async.token, !tt.memdesc<32x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.async.token, !triton_gpu.async.token, !triton_gpu.async.token
    }
    %90 = triton_gpu.async_wait  {num = 0 : i32}
    triton_gpu.local_dealloc %43 : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %44 : !tt.memdesc<2x32x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %91 = tt.splat %arg7 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %92 = arith.cmpi slt, %12, %91 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %93 = tt.load %32, %92, %cst_2 : tensor<32x!tt.ptr<f16>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %94 = arith.sitofp %arg4 : i32 to f32
    %95 = tt.splat %94 : f32 -> tensor<32x32xf32, #mma>
    %96 = arith.mulf %89#0, %95 : tensor<32x32xf32, #mma>
    %97 = arith.sitofp %arg5 : i32 to f16
    %98 = tt.splat %97 : f16 -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %99 = arith.mulf %93, %98 : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %100 = tt.expand_dims %99 {axis = 0 : i32} : tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf16, #mma>
    %101 = arith.extf %100 : tensor<1x32xf16, #mma> to tensor<1x32xf32, #mma>
    %102 = tt.broadcast %101 : tensor<1x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %103 = arith.addf %96, %102 : tensor<32x32xf32, #mma>
    %104 = arith.truncf %103 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %105 = tt.splat %arg11 : i32 -> tensor<32x1xi32, #blocked>
    %106 = arith.muli %105, %13 : tensor<32x1xi32, #blocked>
    %107 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %108 = tt.addptr %107, %106 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %109 = tt.broadcast %108 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %110 = tt.addptr %109, %27 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %111 = arith.andi %37, %40 : tensor<32x32xi1, #blocked>
    %112 = triton_gpu.convert_layout %104 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked>
    tt.store %110, %112, %111 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

