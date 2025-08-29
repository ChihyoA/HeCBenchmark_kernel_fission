#!/bin/bash

# Arrays of values to test
OPT_N_VALUES=(256 512 1024 2048 4096 10000 20000 25000 30000 40000 50000 100000 200000 300000 400000 500000 800000 1000000)
NUM_ITERATIONS_VALUES=(1 10 100 128 256 512 1024)
THREADS_PER_BLOCK_VALUES=(128 256 384 512 1024)

# Output files for each application
OUTPUT_ORIGINAL="blackscholes_results_trueMono_V3.csv"
OUTPUT_SPLIT="blackscholes_splitWithTweak_results_V3.csv"

# Create headers for both files
echo "OPT_N,NUM_ITERATIONS,ThreadsPerBlock,Time(ms)" > $OUTPUT_ORIGINAL
echo "OPT_N,NUM_ITERATIONS,ThreadsPerBlock,Time(ms)" > $OUTPUT_SPLIT

# Run original version
for opt_n in "${OPT_N_VALUES[@]}"; do
    for num_iter in "${NUM_ITERATIONS_VALUES[@]}"; do
        for tpb in "${THREADS_PER_BLOCK_VALUES[@]}"; do
                echo "Running BlackScholes with OPT_N=$opt_n NUM_ITERATIONS=$num_iter TPB=$tpb"
                
                # Run and capture output
                result=$(./BlackScholesMonolithic $opt_n $num_iter $tpb 0 | grep "BlackScholesGPU() time (Total)")
                
                # Extract time value
                time=$(echo $result | awk -F': ' '{print $2}' | awk '{print $1}')
                
                # Save to original version CSV
                echo "$opt_n,$num_iter,$tpb,$time" >> $OUTPUT_ORIGINAL
        done
    done
done

# Run split kernel version
for opt_n in "${OPT_N_VALUES[@]}"; do
    for num_iter in "${NUM_ITERATIONS_VALUES[@]}"; do
        for tpb in "${THREADS_PER_BLOCK_VALUES[@]}"; do
                echo "Running BlackScholesSplit with OPT_N=$opt_n NUM_ITERATIONS=$num_iter TPB=$tpb"
                
                # Run and capture output
                result=$(./BlackScholesKernelSplitTweak $opt_n $num_iter $tpb 0 | grep "BlackScholesGPU() time (Total)")
                
                # Extract time value
                time=$(echo $result | awk -F': ' '{print $2}' | awk '{print $1}')
                
                # Save to split version CSV
                echo "$opt_n,$num_iter,$tpb,$time" >> $OUTPUT_SPLIT
        done
    done
done

echo "Benchmarking complete. Results saved to $OUTPUT_ORIGINAL and $OUTPUT_SPLIT"

