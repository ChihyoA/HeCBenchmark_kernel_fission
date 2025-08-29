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


# program = main_cupti

# source = main_cupti.cu

program = main

source = main.cu

#program = main_split

#source = main_split.cu

# program = main_split_nways

# source = main_split_nways.cu

# program = main_split_nways_with_sync

# source = main_split_nways_with_sync.cu

#program = main_expand

#source = main_expand.cu

# program = main_expand2

# source = main_expand2.cu

# program = main_expand2_nways

# source = main_expand2_nways.cu

#program = main_flat

#source = main_flat.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
HEC_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++14 -arch=$(ARCH) --ptxas-options=-v -lcupti

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

%.o: %.cu Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 10000
run_no_build:
	./$(program) 10000

run_nsys:
#nsys profile -o H100_vanila_high_res5 --gpu-metrics-device=all --gpu-metrics-frequency=100000 --cuda-graph-trace=node 	./$(program) 10
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=100000 --cuda-graph-trace=node 	./$(program) 10
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=100000 --cuda-graph-trace=node 	./$(program) 10

run_ncu_flat:
	ncu -o ncu_flat --set full ./$(program) 10000

run_ncu:
# ncu -o ncu_report --set full ./main 10000
#ncu -o RTX6000_ncu_report_vanila --set full ./$(program) 2
	ncu -o A100_ncu_report_vanila --set full ./$(program) 2
