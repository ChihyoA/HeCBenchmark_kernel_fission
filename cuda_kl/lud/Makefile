all: lud_cuda 

lud_cuda:
	cd cuda; make

clang:
	cd cuda; make clang

clean:
	cd cuda; make clean

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	cuda/lud_cuda -s 1024 -v
