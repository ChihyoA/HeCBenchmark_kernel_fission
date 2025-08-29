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

# program = main_pipeline

# source = main_pipeline.cu

# program = main_pipeline_sync

# source = main_pipeline_sync.cu

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
  HEC_NVCC_FLAGS += -g
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

%.o: %.cu kernels_pipeline.cu loopback.h Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 0 100

run_no_build:
	./$(program) 0 100

run_nsys:
	nsys profile -o H100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 0 100

run_ncu:
	ncu -o H100_ncu_report_vanila  --set full ./$(program) 0 2