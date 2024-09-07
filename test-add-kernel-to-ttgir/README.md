# pipeline 
1. add_kernel.ttir -> coalesce.mlir
2. coalesce.mlir -> optimize-thread.mlir
3. optimize-thrad.mlir -> remove-layout.mlir
4. remove-layout.mlir == add_kernel.ttgir
