import pandas as pd

def compare_kernels(monolithic_file, split_file, output_file):
    mono_df = pd.read_csv(monolithic_file)
    split_df = pd.read_csv(split_file)
    
    # Keep original columns and add group
    mono_df['group'] = ['A' if m < s else 'B' 
                       for m, s in zip(mono_df['time'], split_df['time'])]
    
    # Ensure column order
    columns = ['size', 'blocks', 'times', 'time', 'group']
    mono_df[columns].to_csv(output_file, index=False)

if __name__ == "__main__":
    compare_kernels('benchmark_monolithic.csv', 
                   'benchmark_kernel_split_tweak.csv',
                   'comparison_results.csv')