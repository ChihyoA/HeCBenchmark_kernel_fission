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

# program = motionsim_original

# source = motionsim_original.cu

program = motionsim_kernel-split-combined

source = motionsim_kernel-split-combined.cu


obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
HEC_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++14 -Ddfloat=double -Ddlong=int

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

$(program): $(obj)
	$(LINKER) $(obj) -o $@ $(LINKER_FLAGS)

%.o: %.cu
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) result

run: $(program)
	./$(program) 2000 100 21 147456

run_no_build:
	./$(program) 2000 100 21 147456

run_nsys:
	nsys profile -o H100_split_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 2000 100 21 147456

run_ncu_split:
	ncu -o H100_ncu_report_vanila --set full ./$(program) 2000 2 21 147456

run_ncu:
	ncu -o H100_ncu_report_split --set full ./$(program) 2000 2 21 147456