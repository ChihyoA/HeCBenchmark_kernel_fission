import os
import subprocess
import itertools
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NVIDIA Nsight Compute metrics collection.")
    parser.add_argument("executable", type=str, help="Path to the executable to profile.")
    return parser.parse_args()

def create_output_directory(base_dir, executable_name):
    """Create an output directory specific to the executable."""
    # Extract executable name without path or extension
    executable_base = os.path.basename(executable_name).split('.')[0]
    output_dir = os.path.join(base_dir, executable_base)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def construct_ncu_command(executable, N, threads_per_block, iterations, metrics):
    """Construct the Nsight Compute (ncu) command."""
    return [
        "ncu", "--metrics", ",".join(metrics), "--csv",
        executable, str(N), str(threads_per_block), str(iterations), "0"
    ]

def run_ncu_and_save_metrics(executable, output_dir, N, iterations, threads_per_block, category, metrics):
    """
    Run Nsight Compute for a given configuration and save metrics to a CSV file.
    """
    csv_file = f"{output_dir}/{category}_N{N}_Iter{iterations}_Threads{threads_per_block}.csv"
    command = construct_ncu_command(executable, N, threads_per_block, iterations, metrics)
    
    try:
        print(f"Running {category} metrics for N={N}, Iterations={iterations}, Threads={threads_per_block}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        with open(csv_file, "w") as file:
            file.write(result.stdout)
        print(f"Saved results to {csv_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error profiling N={N}, Iterations={iterations}, Threads={threads_per_block}")
        print(e.output)

def main():
    # Parse arguments
    args = parse_arguments()
    executable = args.executable

    # Configuration values
    OPT_N_VALUES = [256, 512, 1024, 2048, 4096, 10000, 20000, 25000, 30000, 40000, 50000, 100000, 200000, 300000, 400000, 500000, 800000, 1000000, 2000000, 10000000]
    NUM_ITERATIONS_VALUES = [1, 10, 100, 128, 256, 512, 1024]
    THREADS_PER_BLOCK_VALUES = [128, 256, 384, 512, 1024]

    metrics_categories = {
        "Resource_Utilization": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__registers_per_thread",
            "sm__inst_executed.avg.pct_of_peak_sustained",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_launched.sum",
            "sm__pipe_stall_other.avg.pct_of_peak_sustained",
            "sm__issue_active.avg.pct_of_peak_sustained",
            "launch__registers_per_thread",
            "launch__thread_count",
            "launch__occupancy"
        ],
        "Memory_Transfer": [
            "dram__bytes.sum",
            "lts__t_sectors_hit.sum",
            "lts__t_sectors_miss.sum",
        ],
        "Pipeline_Divergence": [
            "sm__pipe_fma_cycles_active",
            "sm__pipe_shared_cycles_active"
        ],
        "Kernel_Specific": [
            "sm__inst_executed.sum",
        ]
    }

    # Create a specific output directory for the executable
    base_output_dir = "./ncu_metric_outputs"
    output_dir = create_output_directory(base_output_dir, executable)

    # Iterate over all configurations
    for N, iterations, threads_per_block in itertools.product(OPT_N_VALUES, NUM_ITERATIONS_VALUES, THREADS_PER_BLOCK_VALUES):
        for category, metrics in metrics_categories.items():
            run_ncu_and_save_metrics(executable, output_dir, N, iterations, threads_per_block, category, metrics)

if __name__ == "__main__":
    main()

