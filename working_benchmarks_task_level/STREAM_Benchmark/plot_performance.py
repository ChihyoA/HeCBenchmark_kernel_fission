import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
kernel_split = pd.read_csv('benchmark_kernel_split_tweak.csv')
monolithic = pd.read_csv('benchmark_monolithic.csv')

# NVIDIA green color
NVIDIA_GREEN = '#76B900'

def create_normalized_chart(size, kernel_split_df, monolithic_df):
    plt.figure(figsize=(12, 6))
    
    # Filter data for the given size
    ks_data = kernel_split_df[kernel_split_df['size'] == size]
    mono_data = monolithic_df[monolithic_df['size'] == size]
    
    # Group by blocks and times to get average performance
    results = []
    
    for (blocks, times), ks_group in ks_data.groupby(['blocks', 'times']):
        mono_group = mono_data[(mono_data['blocks'] == blocks) & 
                              (mono_data['times'] == times)]
        
        if not mono_group.empty:
            ks_time = ks_group['time'].mean()
            mono_time = mono_group['time'].mean()
            normalized = ks_time / mono_time
            results.append({
                'blocks': blocks,
                'iterations': times,
                'normalized': normalized,
                'monolithic': 1.0  # baseline
            })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('normalized')
    
    # Take top 10 best performing configurations
    plot_data = results_df.head(10)
    
    # Plotting
    x = np.arange(len(plot_data))
    width = 0.35
    
    plt.bar(x - width/2, plot_data['monolithic'], width, 
            label='Monolithic (Baseline)', color='black', alpha=0.9)
    plt.bar(x + width/2, plot_data['normalized'], width,
            label='Kernel Split', color=NVIDIA_GREEN, alpha=0.9)
    
    plt.ylabel('Normalized Time (Monolithic = 1.0)')
    plt.title(f'Input Size: {size:,} elements - Normalized Performance (lower is better)')
    
    # Create more descriptive x-axis labels
    labels = [f'Blocks: {row.blocks}\nIterations: {row.iterations}' 
              for _, row in plot_data.iterrows()]
    plt.xticks(x, labels, rotation=45, ha='right')
    
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add some padding at the bottom to prevent label cutoff
    plt.subplots_adjust(bottom=0.2)
    
    # Save the figure
    plt.savefig(f'performance_comparison_{size}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create separate charts for each size
sizes = [40000, 100000, 256000, 512000]
for size in sizes:
    create_normalized_chart(size, kernel_split, monolithic)