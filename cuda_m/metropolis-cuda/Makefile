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

program = main
source = main.cu utils.h kernel_metropolis.h kernel_prng.h kernel_reduction.h  
obj = main.o utils.o heap.o
mainfile = main.o

# program = main_pipeline
# source = main_pipeline.cu utils.h kernel_metropolis_pipeline.h kernel_prng.h kernel_reduction.h  
# obj = main_pipeline.o utils.o heap.o
# mainfile = main_pipeline.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
HEC_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++14

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

$(program): $(obj)
	$(LINKER) $(obj) -o $@ $(LINKER_FLAGS)

heap.o: heap.cu heap.h
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

utils.o: utils.cu heap.h utils.h
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

$(mainfile): $(source)
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat

dbg: $(program)
	./$(program) -l 32 11 -t 4.7 0.1 -a 1 1 1 1 -h 1.0 -z 7919

run: $(program)
	./$(program) -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
run_no_build:
	./$(program) -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 	./$(program) -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
