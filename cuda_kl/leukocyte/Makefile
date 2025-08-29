all: CUDA/leukocyte 

CUDA/leukocyte:
	cd CUDA; make

clean:
	cd CUDA; make clean
	rm -f meschach_lib/meschach.a

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	./CUDA/leukocyte  ../../data/leukocyte/testfile.avi 5 
