#!/bin/bash

# Define the list of benchmark folders
benchmark_dirs=(
  #"cuda_b/bn-cuda"
  #"cuda_c/chemv-cuda"
  #"cuda_de/dense-embedding-cuda"
  #"cuda_fg/f16max-cuda"
  #"cuda_hij/jaccard-cuda"
  #"cuda_m/mixben[ch-cuda"
  #"cuda_nop/page-rank-cuda"
  #"cuda_qr/qrg-cuda"
  #"cuda_s/spm-cuda"
  #"cuda_tz/vote-cuda"
  "cuda_a/adam-cuda"
  "cuda_s/simpleSpmv-cuda"
)

# Loop through each directory and run make run_ncu
for dir in "${benchmark_dirs[@]}"; do
  echo "Entering directory: $dir"
  if [ -d "$dir" ]; then
    pushd "$dir" > /dev/null
    rm ncu_report.ncu-rep
    rm split_ncu_report.ncu-rep
    # if ncu_report.ncu-rep does not exist
    if [ ! -f ncu_report.ncu-rep ]; then    
      echo "Running make run_ncu in $dir"
      make run_ncu &
      make_pid=$!
      # wait for 15 min
      echo "Waiting for 15 minutes..."
      sleep 900
      #kill the job
      echo "Killing the job..."
      kill -SIGINT $make_pid
    fi
    # if split_ncu_report.ncu-rep does not exist
    if [ ! -f split_ncu_report.ncu-rep ]; then
      echo "Running make run_ncu_split in $dir"
      make run_ncu_split &
      make_split_pid=$!
      # wait for 15 min
      echo "Waiting for 15 minutes..."
      sleep 900
      #kill the job
      echo "Killing the job..."
      kill -SIGINT $make_pid
    fi
    popd > /dev/null
  else
    echo "Directory $dir does not exist!"
  fi
done

