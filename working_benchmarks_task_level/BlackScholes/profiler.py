import os
import subprocess
import itertools
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NVIDIA Nsight Compute metrics collection.")
    parser.add_argument("executable", type=str, help="Path to the executable to profile.")
    parser.add_argument("--n-values", nargs="+", type=int, 
                       default=[256, 512, 1024, 2048, 4096, 10000, 20000, 25000, 30000, 40000, 50000, 
                               100000, 200000, 300000, 400000, 500000, 800000, 1000000, 2000000, 10000000],
                       help="List of N values to test")
    parser.add_argument("--iterations", nargs="+", type=int,
                       default=[1, 10, 100, 128, 256, 512, 1024],
                       help="List of iteration counts to test")
    parser.add_argument("--threads", nargs="+", type=int,
                       default=[128, 256, 384, 512, 1024],
                       help="List of thread counts per block to test")
    return parser.parse_args()

def create_output_directory(base_dir, executable_name):
    """Create an output directory specific to the executable."""
    executable_base = os.path.basename(executable_name).split('.')[0]
    output_dir = os.path.join(base_dir, executable_base)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_ncu_output(output_text):
    """Process NCU output and extract just the metrics data."""
    lines = output_text.split('\n')
    data_lines = []
    in_data_section = False
    headers = None

    for line in lines:
        # Skip empty lines and NCU output lines
        if not line.strip() or line.startswith('=='):
            continue

        # Try to identify the header line
        if not headers and ('Kernel Name' in line or 'ID' in line):
            headers = line
            data_lines.append(line)
            in_data_section = True
            continue

        # If we're in the data section, add the line
        if in_data_section and line.strip() and not line.startswith('=='):
            data_lines.append(line)

    return '\n'.join(data_lines) if data_lines else None

def run_ncu_and_save_metrics(executable, output_dir, N, iterations, threads_per_block, category, metrics):
    """Run Nsight Compute and save only the actual metrics data."""
    csv_file = os.path.join(
        output_dir, 
        f"{category}_N{N}_I{iterations}_T{threads_per_block}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    command = [
        "ncu",
        "--metrics", ",".join(metrics),
        "--csv",
        executable,
        str(N),
        str(threads_per_block),
        str(iterations),
        "0"
    ]
    
    try:
        print(f"\nRunning {category} metrics for N={N}, Iterations={iterations}, Threads={threads_per_block}")
        print(f"Command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Process the output
        clean_output = process_ncu_output(result.stdout)
        
        if clean_output:
            # Write the cleaned data to file
            with open(csv_file, 'w') as f:
                f.write(clean_output)
            print(f"Saved cleaned results to {csv_file}")
            return True
        else:
            print("No valid data found in the output")
            print("Raw output first 500 chars:")
            print(result.stdout[:500])
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running profiler: {str(e)}")
        if e.output:
            print("Error output:")
            print(e.output)
        return False

def main():
    args = parse_arguments()
    
    # Define metrics categories
    metrics_categories = {
        "Resource_Utilization": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__registers_per_thread",
            "sm__inst_executed.avg.pct_of_peak_sustained",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_launched.sum",
            "sm__pipe_stall_other.avg.pct_of_peak_sustained",
            "sm__issue_active.avg.pct_of_peak_sustained",
            "launch__thread_count",
            "launch__occupancy"
        ],
        "Memory_Performance": [
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "lts__t_bytes.sum",
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st_hit_rate.pct"
        ],
        "Pipeline_Efficiency": [
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained",
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained",
            "sm__inst_executed.avg.pct_of_peak_sustained",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained",
            "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained"
        ],
        "Compute_Throughput": [
            "sm__inst_executed_pipe_alu.avg",
            "sm__inst_executed_pipe_fma.avg",
            "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp64_pred_on.sum"
        ]
    }
    
    # Create output directory
    base_output_dir = "./ncu_metric_outputs"
    output_dir = create_output_directory(base_output_dir, args.executable)
    
    # Iterate over all configurations
    for N, iterations, threads_per_block in itertools.product(args.n_values, args.iterations, args.threads):
        for category, metrics in metrics_categories.items():
            run_ncu_and_save_metrics(args.executable, output_dir, N, iterations, threads_per_block, category, metrics)

if __name__ == "__main__":
    main()


