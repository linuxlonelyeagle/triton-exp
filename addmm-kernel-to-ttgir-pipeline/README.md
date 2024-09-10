# pipeline 
## addmm_kernel.ttir -> ttgir
`triton-opt addmm_kernel.ttir -convert-triton-to-tritongpu="target=cuda:80" -o triton-gpu.mlir`
## run coalesce
`triton-opt triton-gpu.mlir -tritongpu-coalesce -o coalesce.mlir`
## f32dot
`triton-opt coalesce.mlir -tritongpu-F32DotTC -o f32dottc.mlir`
## remove layout
`triton-opt f32dottc.mlir -tritongpu-remove-layout-conversions -o remove-layout.mlir`
## thread-locality
`triton-opt remove-layout.mlir -tritongpu-optimize-thread-locality -o thread-locality.mlir`
## accelerate matmul
`triton-opt thread-locality.mlir -tritongpu-accelerate-matmul -o accelerate-matmul.mlir`
## remove layout
`triton-opt accelerate-matmul.mlir -tritongpu-remove-layout-conversions -o remove-layout.mlir`
## optimize-dot
`triton-opt remove-layout_1.mlir -tritongpu-optimize-dot-operands -o optimize-dot.mlir`
## cse
`triton-opt optimize-dot.mlir -cse -o cse.mlir`
## combine-tensor
`triton-opt cse.mlir -o tritongpu-combine-tensor-select-and-if -o combine-tensor.mlir`
## pipeline
`triton-opt combine-tensor.mlir -tritongpu-pipeline -o pipeline.mlir`
