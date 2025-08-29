include ../../common/make.config
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_90

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

# program = main_split

# source = main_split.cu

# program = main_split_nways

# source = main_split_nways.cu

# program = main_split_nways_correct

# source = main_split_nways_correct.cu


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

%.o: %.cu Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 100000 100

run_no_build:
	./$(program) 100000 100

run_nsys:
#nsys profile -o H100_split_8ways --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./$(program) 100000 100
	nsys profile -o H100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 100000 100

run_ncu:
#ncu -o pipeline_ncu_report --set launchstats ./$(program) 100000 100
	ncu -o H100_ncu_report_vanila --kernel-name ForceCalculationKernel --set full ./$(program) 100000 2

run_ncu_split:
	ncu -o pipeline_ncu_report --set launchstats ./main 100000 100