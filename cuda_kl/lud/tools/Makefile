gen_input: gen_input.c
	gcc -o gen_input -O3 gen_input.c -fopenmp

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	gcc -o gen_input -O3 gen_input.c -fopenmp
