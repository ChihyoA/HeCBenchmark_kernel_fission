include ../../common/make.config
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
OPTIMIZE  = yes
DEBUG     = no
#ARCH      = sm_90 #H100
# ARCH      = sm_80 #A100
ARCH	  = sm_75 #RTX 6000

#===============================================================================
# Program name & source code list
#===============================================================================


# program = main_split_nways_with_sync

# source = main_split_nways_with_sync.cu

# program = main_split_nways

# source = main_split_nways.cu

#program = main_split_eventwait

#source = main_split_eventwait.cu

# program = main_split

# source = main_split.cu

#program = main_expand2

#source = main_expand.cu

# program = main_flat	

# source = main_flat.cu

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
#nsys profile -o H100_split_45ways_sync --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./$(program) result 10
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./$(program) result 10
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=10000 --cuda-graph-trace=node 	./$(program) result 10

run_ncu_split:
#ncu -o split_ncu_report --set launchstats ./$(program) result 10
	ncu -o split_ncu_report --set full ./$(program) result 10

run_ncu:
#ncu -o ncu_report --set launchstats ./main result 10
# ncu -o H100_ncu_report_split_45ways --kernel-name "genScoreKernel" --set full ./main result 2
#ncu -o RTX6000_ncu_report_vanila --kernel-name "genScoreKernel" --set full ./$(program) result 2
	ncu -o A100_ncu_report_vanila --kernel-name "genScoreKernel" --set full ./$(program) result 2

