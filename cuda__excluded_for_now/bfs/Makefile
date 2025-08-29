include ../../common/make.config

#CC := $(CUDA_DIR)/bin/nvcc
#
#INCLUDE := $(CUDA_DIR)/include
#

SRC = bfs.cu
OBJ = bfs.o
EXE = bfs.out

$(EXE): $(OBJ)
	$(LINKER) $(OBJ) -o $(EXE) $(LINKER_FLAGS)

$(OBJ): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -I../util


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *.o

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	rm -f $(EXE) $(EXE).linkinfo result.txt *.o
