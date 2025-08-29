#!/bin/bash

# Define the list of benchmark folders
# benchmark_dirs=(
#   "cuda_b/bn-cuda"
#   "cuda_c/chemv-cuda"
#   "cuda_d/dense-embedding-cuda"
#   "cuda_fg/f16max-cuda"
#   "cuda_hij/jaccard-cuda"
#   "cuda_m/mixbench-cuda"
#   "cuda_nop/page-rank-cuda"
#   "cuda_qr/qrg-cuda"
#   "cuda_s/spm-cuda"
#   "cuda_tz/vote-cuda"
#   "cuda_a/adam-cuda"
#   "cuda_s/simpleSpmv-cuda"
# )
benchmark_dirs=(
  # all benchmarks in working_benchmarks_loop, working_benchmarks_atomics, working_benchmarks_pipeline, working_benchmarks_task_level
  
  # working_benchmarks_loop
  "working_benchmarks_loop/accuracy-cuda"
  # "working_benchmarks_loop/bh-cuda"
  "working_benchmarks_loop/bn-cuda"
  "working_benchmarks_loop/chemv-cuda"
  "working_benchmarks_loop/dense-embedding-cuda"
  "working_benchmarks_loop/f16max-cuda"
  "working_benchmarks_loop/jaccard-cuda"
  "working_benchmarks_loop/matrix-rotate-cuda"
  "working_benchmarks_loop/mixbench-cuda"
  # "working_benchmarks_loop/page-rank-cuda"
  "working_benchmarks_loop/qrg-cuda"
  "working_benchmarks_loop/spm-cuda"
  # "working_benchmarks_loop/vote-cuda"
  
  # working_benchmarks_atomics
  "working_benchmarks_atomics/adam-cuda"
  "working_benchmarks_atomics/clenergy-cuda"
  "working_benchmarks_atomics/simpleSpmv-cuda (meaningless_loop)"
  
  # working_benchmarks_pipeline
  # "working_benchmarks_pipeline/adv-cuda"
  # "working_benchmarks_pipeline/fdtd3d-cuda"
  # "working_benchmarks_pipeline/fhd-cuda"
  # "working_benchmarks_pipeline/lda-cuda"
  # "working_benchmarks_pipeline/loopback-cuda"
  
  # working_benchmarks_task_level
  # "working_benchmarks_task_level/BlackScholes"
  # "working_benchmarks_task_level/STREAM_Benchmark"
  # "working_benchmarks_task_level/ace-cuda"
  # "working_benchmarks_task_level/adv-cuda"
  # "working_benchmarks_task_level/particle-diffusion-cuda"
)

# Loop through each directory and run make run_ncu
for dir in "${benchmark_dirs[@]}"; do
  echo "Entering directory: $dir"
  if [ -d "$dir" ]; then
    pushd "$dir" > /dev/null
    echo "Running make run_ncu in $dir"
    make clean
    make
    make run_nsys
    popd > /dev/null
  else
    echo "Directory $dir does not exist!"
  fi
done