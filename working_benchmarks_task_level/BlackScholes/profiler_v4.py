#!/usr/bin/env python3
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import List, Dict
import json

class GPUProfiler:
    def __init__(self, app_path: str, output_dir: str = "profiling_results"):
        self.app_path = app_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / f"profiling_{self.timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Metrics to collect
        self.metrics = [
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
            "lts__t_sectors_op_write.sum"
        ]

    def run_profiling(self, config_name: str, args: List[str], num_runs: int = 5) -> None:
        """Run profiling for a specific configuration multiple times."""
        self.logger.info(f"Starting profiling for configuration: {config_name}")
        
        results_dir = self.output_dir / config_name / self.timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        for run in range(1, num_runs + 1):
            self.logger.info(f"Run {run}/{num_runs}")
            output_file = results_dir / f"run_{run}.csv"
            
            # Construct ncu command
            cmd = [
                "ncu",
                "--metrics", ",".join(self.metrics),
                "--csv",
                "--output-path", str(output_file),
                "--replay-mode application",  # Better for kernel profiling
                "--profile-from-start 1",  # Start profiling from the beginning
                self.app_path
            ] + args

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                self.logger.info(f"Completed run {run}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error in run {run}: {e}")
                self.logger.error(f"Command output: {e.output}")

    def process_results(self, config_name: str) -> pd.DataFrame:
        """Process results for a configuration and generate statistics."""
        results_dir = self.output_dir / config_name / self.timestamp
        all_data = []

        for csv_file in results_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                # Extract relevant columns
                metrics_data = df[["Kernel Name", "Metric Name", "Metric Value", "Metric Unit"]]
                all_data.append(metrics_data)
            except Exception as e:
                self.logger.error(f"Error processing {csv_file}: {e}")

        if not all_data:
            self.logger.error(f"No data found for {config_name}")
            return None

        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Calculate statistics
        stats = combined_data.groupby(["Kernel Name", "Metric Name", "Metric Unit"]).agg({
            "Metric Value": ["mean", "std", "min", "max"]
        }).round(3)

        # Save statistics
        stats_file = self.output_dir / f"stats_{config_name}_{self.timestamp}.xlsx"
        stats.to_excel(stats_file)
        
        return stats

    def generate_comparison(self, configs: List[str]) -> None:
        """Generate comparison between different configurations."""
        all_stats = {}
        for config in configs:
            stats = self.process_results(config)
            if stats is not None:
                all_stats[config] = stats

        if not all_stats:
            self.logger.error("No data available for comparison")
            return

        # Create comparison Excel file
        comparison_file = self.output_dir / f"comparison_{self.timestamp}.xlsx"
        with pd.ExcelWriter(comparison_file) as writer:
            for config, stats in all_stats.items():
                stats.to_excel(writer, sheet_name=config)

            # Create comparison sheet
            comparison_df = pd.DataFrame()
            for config, stats in all_stats.items():
                stats_flat = stats["Metric Value"]["mean"].reset_index()
                stats_flat.columns = ["Kernel Name", "Metric Name", "Metric Unit", config]
                if comparison_df.empty:
                    comparison_df = stats_flat
                else:
                    comparison_df = comparison_df.merge(
                        stats_flat[["Kernel Name", "Metric Name", config]],
                        on=["Kernel Name", "Metric Name"],
                        how="outer"
                    )
            
            comparison_df.to_excel(writer, sheet_name="Comparison")

def main():
    parser = argparse.ArgumentParser(description="GPU Kernel Profiling Tool")
    parser.add_argument("--app_path", required=True, help="Path to the application")
    parser.add_argument("--output_dir", default="profiling_results", help="Output directory")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per configuration")
    args = parser.parse_args()

    profiler = GPUProfiler(args.app_path, args.output_dir)

    # Example configurations
    configs = {
        "monolithic": ["--kernel-type=monolithic"],
        "split": ["--kernel-type=split"]
    }

    # Run profiling for each configuration
    for config_name, config_args in configs.items():
        profiler.run_profiling(config_name, config_args, args.num_runs)

    # Generate comparison
    profiler.generate_comparison(list(configs.keys()))

if __name__ == "__main__":
    main()




