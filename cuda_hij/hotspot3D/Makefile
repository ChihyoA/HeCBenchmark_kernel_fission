include ../../common/make.config

CC := $(NVCC)

INCLUDE := $(CUDA_DIR)/include

SRC = 3D.cu
OBJ = 3D.o
EXE = 3D

OUTPUT = *.out

$(EXE): $(OBJ)
	$(LINKER) $(OBJ) -o $(EXE) $(LINKER_FLAGS)

$(OBJ): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -I../util

enum: $(SRC)
	$(NVCC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(NVCC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(NVCC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) *.o

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out 1