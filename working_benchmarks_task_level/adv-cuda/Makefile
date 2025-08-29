include ../../common/make.config
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60

#===============================================================================
# Program name & source code list
#===============================================================================

#program = main

#source = main.cu

program = main_combined

source = main_combined.cu


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
	./$(program) 7 15 8000 100

run_no_build:
	./$(program) 7 15 8000 100

run_nsys:
	nsys profile -o vanila2 --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	./$(program) 7 15 256 100

run_ncu_split:
	ncu -o split_ncu_report --set full ./$(program) 7 15 8000 100

run_ncu:
	ncu -o ncu_report --set full ./main 7 15 8000 100