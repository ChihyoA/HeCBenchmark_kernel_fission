#!/usr/bin/env python3
import os
import subprocess
import itertools
import argparse
import logging
from datetime import datetime
from typing import List, Dict

def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger('ncu_profiler')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, f'profiling_{datetime.now():%Y%m%d_%H%M%S}.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NVIDIA Nsight Compute metrics collection.")
    parser.add_argument("executable", type=str, help="Path to the executable to profile")
    parser.add_argument("--output-dir", type=str, default="./ncu_metric_outputs",
                      help="Base output directory for results")
    parser.add_argument("--n-values", type=int, nargs="+",
                      default=[256, 512, 1024, 2048, 4096, 10000, 20000, 25000, 
                              30000, 40000, 50000, 100000, 250000, 500000, 1000000],
                      help="List of N values to test")
    parser.add_argument("--iterations", type=int, nargs="+",
                      default=[128, 256, 512],
                      help="List of iteration values to test")
    parser.add_argument("--threads", type=int, nargs="+",
                      default=[128, 256, 384, 512, 1024],
                      help="List of threads per block values to test")
    return parser.parse_args()

def create_output_directory(base_dir: str, executable_name: str) -> str:
    executable_base = os.path.basename(executable_name).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{executable_base}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def construct_ncu_command(executable: str, N: int, threads_per_block: int, 
                         iterations: int, metrics: List[str]) -> List[str]:
    return [
        "ncu",
        "--metrics", ",".join(metrics),
        "--csv",
        "--replay-mode", "application",
        "--profile-from-start", "1",
        executable,
        str(N),
        str(iterations),
        str(threads_per_block),
        "0"
    ]

def run_ncu_and_save_metrics(executable: str, output_dir: str, N: int, 
                            iterations: int, threads_per_block: int, 
                            category: str, metrics: List[str], 
                            logger: logging.Logger) -> None:
    csv_file = os.path.join(output_dir, 
                           f"{category}_N{N}_Iter{iterations}_Threads{threads_per_block}.csv")
    
    command = construct_ncu_command(executable, N, threads_per_block, iterations, metrics)
    
    try:
        logger.info(f"Running {category} metrics - N={N}, Iter={iterations}, "
                   f"Threads={threads_per_block}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        with open(csv_file, "w") as f:
            f.write(result.stdout)
        logger.info(f"Results saved to {csv_file}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running profiling: {e}")
        logger.error(f"Command output: {e.output}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    args = parse_arguments()
    output_dir = create_output_directory(args.output_dir, args.executable)
    logger = setup_logging(output_dir)

    metrics_categories = {
        "performance_metrics": [
            "sm__cycles_elapsed.avg",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__occupancy.avg.pct_of_peak_sustained_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "lts__t_sectors_op_read.sum",
            "lts__t_sectors_op_write.sum",
            "l1tex__t_bytes.sum",
            "lts__t_bytes.sum",
            "dram__bytes.sum",
            "launch__thread_count",
            "sm__inst_executed.avg.per_cycle_active",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"
        ]
    }

    # Save metric configuration
    with open(os.path.join(output_dir, "metric_config.txt"), "w") as f:
        for category, metrics in metrics_categories.items():
            f.write(f"\n{category}:\n")
            for metric in metrics:
                f.write(f"  - {metric}\n")

    # Run profiling
    for N, iterations, threads in itertools.product(
            args.n_values, args.iterations, args.threads):
        for category, metrics in metrics_categories.items():
            run_ncu_and_save_metrics(
                args.executable, output_dir, N, iterations, 
                threads, category, metrics, logger
            )

if __name__ == "__main__":
    main()

