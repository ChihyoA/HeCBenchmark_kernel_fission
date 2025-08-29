include ../../common/make.config

CC := $(NVCC)

INCLUDE := $(CUDA_DIR)/include


SRC = srad.cu
OBJ = srad.o
EXE = srad

$(EXE): $(OBJ)
	$(LINKER) $(OBJ) -o $(EXE) $(LINKER_FLAGS)

$(OBJ): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -I../util

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *.o

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	./srad 20480 20480 0 127 0 127 0.5 2
