include ../../common/make.config
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
OPTIMIZE  = yes
DEBUG     = no
#ARCH      = sm_90
# ARCH      = sm_80 # A100
ARCH	  = sm_75 #RTX 6000

#===============================================================================
# Program name & source code list
#===============================================================================

# program = main_pipeline

# source = main.cpp kernels_pipeline.cu utils.cpp

# obj = main.o kernels_pipeline.o utils.o

# program = main_split_atomic

# source = main.cpp kernels_split_atomic.cu utils.cpp

# obj = main.o kernels_split_atomic.o utils.o

# program = main_split_atomic_nways

# source = main.cpp kernels_split_atomic_nways.cu utils.cpp

# obj = main.o kernels_split_atomic_nways.o utils.o

program = main

source = main.cpp kernels.cu utils.cpp

obj = main.o kernels.o utils.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
HEC_GCC_FLGAS := $(NVCC_FLAGS) -std=c++14
HEC_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++14 -arch=$(ARCH) --ptxas-options=-v 

# Linker Flags
LINKER_FLAGS += 

# Debug Flags
ifeq ($(DEBUG),yes)
  HEC_NVCC_FLAGS += -g -DDEBUG
  LINKER_FLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  HEC_NVCC_FLAGS += 
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(LINKER) $(obj) -o $@ $(LINKER_FLAGS)

%.o: %.cu mv.h Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

%.o: %.cpp mv.h Makefile
	$(CXX) $(HEC_GCC_FLGAS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 16777216 10240 100
#./$(program) 128 32 1
run_no_build:
	./$(program) 16777216 10240 100

run_nsys:
#nsys profile -o H100_vanila_high_res2 --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 16777216 10240 100
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 16777216 10240 100
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 16777216 10240 100

run_ncu_split:
	ncu -o pipeline_ncu_report  --set full	./$(program) 16777216 10240 100

run_ncu:
#ncu -o H100_ncu_report_vanila --kernel-name "mv_csr"  --set full ./$(program) 16777216 10240 2
#ncu -o RTX6000_ncu_report_vanila --kernel-name "mv_csr"  --set full ./$(program) 16777216 10240 2
	ncu -o A100_ncu_report_vanila --kernel-name "mv_csr"  --set full ./$(program) 16777216 10240 2