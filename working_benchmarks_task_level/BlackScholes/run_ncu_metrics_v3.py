import os
import subprocess
import itertools
import argparse
from datetime import datetime
import pandas as pd
import csv
import io

def clean_ncu_output(output_text):
    """Clean NCU output to get only the metrics data."""
    # Find the start of the actual CSV data (after the header fluff)
    lines = output_text.split('\n')
    
    # Find the line that starts with "==PROF== Disconnected"
    # and the line that contains the actual CSV headers
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID","Process ID"') or line.startswith('"Kernel Name"'):
            start_idx = i
            break
    
    if start_idx is not None:
        # Join only the relevant lines
        clean_output = '\n'.join(lines[start_idx:])
        return clean_output
    return None

def parse_metrics_to_df(output_text):
    """Parse NCU output into a clean DataFrame."""
    clean_csv = clean_ncu_output(output_text)
    if clean_csv:
        try:
            # Read the cleaned CSV data
            df = pd.read_csv(io.StringIO(clean_csv))
            
            # Drop any empty rows
            df = df.dropna(how='all')
            
            # Remove any rows that don't contain actual data
            df = df[~df.iloc[:, 0].str.startswith("==").fillna(False)]
            
            return df
        except Exception as e:
            print(f"Error parsing CSV data: {e}")
            return None
    return None

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
        print(f"Running {category} metrics for N={N}, Iterations={iterations}, Threads={threads_per_block}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Parse and clean the output
        df = parse_metrics_to_df(result.stdout)
        
        if df is not None:
            # Save only the cleaned data
            df.to_csv(csv_file, index=False)
            print(f"Saved cleaned results to {csv_file}")
            return True
        else:
            print("Failed to parse metrics output")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running profiler: {str(e)}")
        print(e.output)
        return False

def main():
    # Your existing main function code here, but use the new run_ncu_and_save_metrics function
    parser = argparse.ArgumentParser(description="Run NVIDIA Nsight Compute metrics collection.")
    parser.add_argument("executable", type=str, help="Path to the executable to profile.")
    args = parse_arguments()
    
    # Configuration values (you can keep your existing values)
    metrics_categories = {
        "Resource_Utilization": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__registers_per_thread",
            "sm__inst_executed.avg.pct_of_peak_sustained",
            # ... rest of your metrics
        ],
        # ... rest of your categories
    }
    
    # Create output directory
    base_output_dir = "./ncu_metric_outputs"
    output_dir = create_output_directory(base_output_dir, args.executable)
    
    # Your existing iteration code using the new run_ncu_and_save_metrics function
    for N, iterations, threads_per_block in itertools.product(OPT_N_VALUES, NUM_ITERATIONS_VALUES, THREADS_PER_BLOCK_VALUES):
        for category, metrics in metrics_categories.items():
            run_ncu_and_save_metrics(args.executable, output_dir, N, iterations, threads_per_block, category, metrics)

if __name__ == "__main__":
    main()

