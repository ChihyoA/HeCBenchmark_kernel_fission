import subprocess
import itertools

element_sizes = [4096, 10000, 40000, 100000, 256000, 512000, 1000000]
block_sizes = [128, 192, 256, 384, 512, 768, 1024]
ntimes = [10, 20, 50, 100, 10000]

def run_benchmark(executable, n, block_size, ntimes):
   cmd = [executable, '-n', str(n), '-b', str(block_size), '-t', str(ntimes)]
   try:
       result = subprocess.run(cmd, capture_output=True, text=True, check=True)
       for line in result.stdout.split('\n'):
           if "Total Time:" in line:
               time = float(line.split(':')[1].strip().split()[0])
               return time
   except subprocess.CalledProcessError as e:
       print(f"Error: {e}")
       return None

for executable in ['monolithic', 'kernel_split_tweak']:
    with open(f'benchmark_{executable}.csv', 'w') as f:
        f.write('size,blocks,times,time\n')
        for n, block, times in itertools.product(element_sizes, block_sizes, ntimes):
            time = run_benchmark(f'./stream_{executable}', n, block, times)
            f.write(f'{n},{block},{times},{time}\n')