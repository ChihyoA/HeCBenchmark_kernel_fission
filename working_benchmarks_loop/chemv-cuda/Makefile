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

#program = main_split

#source = kernel_split.cu  main_split.cu

# program = main_split_nways

# source = kernel_split_nways.cu  main_split_nways.cu

# program = main_split_nways_with_sync

# source = kernel_split_nways.cu  main_split_nways_with_sync.cu

#program = main_expand2

#source = kernel_expand2.cu  main_expand2.cu

# program = main_expand2_nways

# source = kernel_expand2_nways.cu  main_expand2_nways.cu

program = main

source = kernel.cu  main.cu

#program = main_flat

#source = kernel_flat.cu  main_flat.cu

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
  HEC_NVCC_FLAGS +=  #-fmad=true
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(LINKER) $(obj) -o $@ $(LINKER_FLAGS)

%.o: %.cu kernel.h
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)
run_no_build:
	./$(program)


run_nsys:
#nsys profile -o H100_split_19ways_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)

run_ncu_split:
	ncu -o split_ncu_report --set full ./$(program) 

run_ncu:
# ncu -o H100_ncu_report_vanila --kernel-name "kernel1" --set full ./$(program) 
#ncu -o RTX6000_ncu_report_vanila --kernel-name "kernel1" --set full ./$(program) 
	ncu -o A100_ncu_report_vanila --kernel-name "kernel1" --set full ./$(program) 