
nvcc stream_kernel_split_tweak.cu -o stream_kernel_split_tweak_mark.o

# ./stream_kernel_split_tweak -s -n 10000000 -b 128 -t 10
# ./stream_monolithic -s -n 10000000 -b 128 -t 10

nsys profile -o H100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./stream_kernel_split_tweak -s -n 10000000 -b 128 -t 10

# nsys profile -o H100_split_high_res --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./stream_monolithic -s -n 10000000 -b 128 -t 10