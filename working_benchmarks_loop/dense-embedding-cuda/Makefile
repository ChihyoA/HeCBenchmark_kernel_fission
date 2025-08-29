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

# program = main_split_nways_with_sync

# source = main_split_nways_with_sync.cu

# program = main_split_nways

# source = main_split_nways.cu

#program = main_expand2

#source = main_expand2.cu

#program = main_flat

#source = main_flat.cu

program = main

source = main.cu

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
	./$(program) 100000 256 1000

run_no_build:
	./$(program) 100000 256 1000

run_nsys:
#nsys profile -o H100_vanila_high_res3 --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 100000 256 1000
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 100000 256 1000
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node 	./$(program) 100000 256 1000

run_ncu_split:
	ncu -o split_ncu_report --set full ./$(program) 100000 256 1000

run_ncu:
# ncu -o H100_ncu_report_split_8ways_sync_final --set full ./main 100000 256 2
#ncu -o RTX6000_ncu_report_vanila --set full ./$(program) 100000 256 2
	ncu -o A100_ncu_report_vanila --set full ./$(program) 100000 256 2
