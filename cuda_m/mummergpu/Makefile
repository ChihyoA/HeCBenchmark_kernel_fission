all: mummer 

mummer:
	cd src; make

clean:
	cd src; make clean
	

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	bin/mummergpu ../../data/mummergpu/NC_003997.fna ../../data/mummergpu/NC_003997_q100bp.fna > NC_00399.out
