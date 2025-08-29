#!/bin/bash

# Configuration variables
OUTPUT_DIR="profiling_results"
NUM_RUNS=5  # Number of runs for each configuration
APP_PATH="./your_application"  # Replace with your application path
DATE=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to run profiling
run_profiling() {
    local output_file=$1
    local config_name=$2
    
    # All metrics collected in one group for efficiency
    ncu --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_active,\
        l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__occupancy.avg.pct_of_peak_sustained_active,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        lts__throughput.avg.pct_of_peak_sustained_elapsed,\
        launch__registers_per_thread,\
        launch__shared_mem_per_block_static,\
        sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
        lts__t_sectors_op_read.sum,\
        lts__t_sectors_op_write.sum \
        --csv \
        --target-processes all \
        --output-path "${output_file}" \
        ${APP_PATH} $3  # $3 forwards additional arguments to your application
}

# Function to process results
process_results() {
    local config_name=$1
    echo "Processing results for ${config_name}..."
    
    # Combine all CSV files for this configuration
    echo "Run,Metric,Value" > "${OUTPUT_DIR}/combined_${config_name}.csv"
    for run in $(seq 1 ${NUM_RUNS}); do
        if [ -f "${OUTPUT_DIR}/${config_name}_run${run}.csv" ]; then
            tail -n +2 "${OUTPUT_DIR}/${config_name}_run${run}.csv" | \
            awk -F',' -v run="${run}" '{print run "," $13 "," $15}' >> \
            "${OUTPUT_DIR}/combined_${config_name}.csv"
        fi
    done
}

# Main execution
run_experiment() {
    local config_name=$1
    shift  # Remove first argument
    local args="$@"  # Remaining arguments

    echo "Running experiment: ${config_name}"
    for run in $(seq 1 ${NUM_RUNS}); do
        echo "Run ${run}/${NUM_RUNS}"
        output_file="${OUTPUT_DIR}/${config_name}_run${run}"
        run_profiling "${output_file}" "${config_name}" "${args}"
    done

    process_results "${config_name}"
}

# Example usage:
# 1. Monolithic kernel
run_experiment "monolithic" "--kernel-type=monolithic"

# 2. Split kernel
run_experiment "split" "--kernel-type=split"

# Generate summary statistics using Python
python3 - <<EOF
import pandas as pd
import numpy as np

def process_config(config_name):
    df = pd.read_csv(f"${OUTPUT_DIR}/combined_{config_name}.csv")
    summary = df.groupby('Metric').agg({
        'Value': ['mean', 'std', 'min', 'max']
    }).round(3)
    return summary

# Process both configurations
monolithic_stats = process_config('monolithic')
split_stats = process_config('split')

# Save results
with pd.ExcelWriter('${OUTPUT_DIR}/summary_${DATE}.xlsx') as writer:
    monolithic_stats.to_excel(writer, sheet_name='Monolithic')
    split_stats.to_excel(writer, sheet_name='Split')
EOF

echo "Profiling completed. Results are in ${OUTPUT_DIR}/summary_${DATE}.xlsx"

