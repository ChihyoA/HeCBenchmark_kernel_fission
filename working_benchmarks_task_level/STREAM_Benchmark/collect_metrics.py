import subprocess
import itertools
import os
from datetime import datetime

configs = {
   'sizes': [4096, 10000, 40000, 100000, 256000, 512000, 1000000],
   'blocks': [128, 192, 256, 384, 512, 768, 1024],
   'times': [10, 20, 50, 100, 10000]
}

metrics = {
   'resource_utilization': [
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
   'memory_transfer': [
       "dram__bytes.sum",
       "lts__t_sectors_hit.sum",
       "lts__t_sectors_miss.sum",
   ],
   'kernel_specific': [
       "sm__inst_executed.sum",
   ]
}

def run_benchmark(executable, size, block, times, metric_group, metrics_list):
   output_dir = f"ncu_results_{datetime.now().strftime('%Y%m%d')}/{metric_group}"
   os.makedirs(output_dir, exist_ok=True)
   
   output_file = f"{output_dir}/{executable.split('/')[-1]}_n{size}_b{block}_t{times}.csv"
   
   cmd = ['sudo', '/usr/local/cuda/bin/ncu', '--replay-mode', 'application', '--metrics', ','.join(metrics_list), '--csv',
          executable, '-s', '-n', str(size), '-b', str(block), '-t', str(times)]
   
   try:
       print(f"Running {' '.join(cmd)}")
       result = subprocess.run(cmd, capture_output=True, text=True, check=True)
       
       with open(output_file, 'w') as f:
           f.write(result.stdout)
           
       print(f"Results saved to {output_file}")
       return output_file
   except subprocess.CalledProcessError as e:
       print(f"Error with command: {e}")
       return None

executables = ['./stream_monolithic', './stream_kernel_split_tweak']

# Run for each metric group
for metric_group, metrics_list in metrics.items():
   print(f"\nCollecting {metric_group} metrics...")
   for exe, size, block, times in itertools.product(executables, configs['sizes'], configs['blocks'], configs['times']):
       run_benchmark(exe, size, block, times, metric_group, metrics_list)