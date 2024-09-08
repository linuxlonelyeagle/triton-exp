# pipeline 
1. add_kernel.ttgir -> coalesce.mlir
`triton-opt add_kernel.ttgir -tritongpu-coalesce -o coalesce.mlir`
2. coalesce.mlir -> optimize-thread.mlir
`triton-opt coalesce.mlir -tritongpu-optimize-thread-locality -o optimize-thread.mlir`
3. optimize-thrad.mlir -> remove-layout.mlir
`triton-opt optimize-thread.mlir -tritongpu-remove-layout-conversions -o remove-layout.mlir`
4. remove-layout.mlir == add_kernel.ttgir
