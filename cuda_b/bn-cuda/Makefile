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

program = main_split

source = main_split.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
HEC_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++14

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

%.o: %.cu kernels.cu data45.h
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) result

run: $(program)
	./$(program) result 10
run_no_build:
	./$(program) result 10

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	./$(program) result 10

run_ncu_split:
#ncu -o split_ncu_report --set launchstats ./$(program) result 10
	ncu -o split_ncu_report --set full ./$(program) result 10

run_ncu:
#ncu -o ncu_report --set launchstats ./main result 10
	ncu -o ncu_report --set full ./main result 10

