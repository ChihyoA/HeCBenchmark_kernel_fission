include ../../common/make.config
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
OPTIMIZE  = yes
DEBUG     = no
#ARCH      = sm_90
# ARCH      = sm_80 # A100
ARCH	  = sm_75 # RTX6000

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

# program = main_split

# source = main_split.cu

# program = main_flat

# source = main_flat.cu

# program = main_split_nways

# source = main_split_nways.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
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

%.o: %.cu reference.h Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 8192 10000 10 100

run_no_build:
	./$(program) 8192 10000 10 100

run_nsys:
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=60000 --cuda-graph-trace=node 	./$(program) 8192 10000 10 100
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=60000 --cuda-graph-trace=node 	./$(program) 8192 10000 10 100
#nsys profile -o H100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=60000 --cuda-graph-trace=node 	./$(program) 8192 10000 10 100
#nsys profile -o H200_vanila_high_res2 --gpu-metrics-device=all --gpu-metrics-frequency=60000 --cuda-graph-trace=node 	./$(program) 8192 10000 10 100

run_ncu:
#ncu -o split_ncu_report --set launchstats ./$(program) result 10
# ncu -o H100_ncu_report_vanila --set full ./$(program) 8192 10000 10 2
#ncu -o RTX6000_ncu_report_vanila --set full ./$(program) 8192 10000 10 2
	ncu -o A100_ncu_report_vanila --set full ./$(program) 8192 10000 10 2
#ncu --target-processes all --launch-count 0 --kernel-name accuracy_kernel_split -o ncu_report_split_check --set full ./$(program) 8192 10000 10 2
#ncu --print-summary per-gpu --csv --set full ./$(program) 8192 10000 10 2
#ncu --launch-count 10 --kernel-name accuracy_kernel_split -o ncu_report_split_check --set full ./$(program) 8192 10000 10 2