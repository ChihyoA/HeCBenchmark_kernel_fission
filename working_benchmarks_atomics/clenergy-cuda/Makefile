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

program = clenergy

source = clenergy.cu WKFUtils.cu

# program = clenergy_atomics

# source = clenergy_atomics.cu WKFUtils.cu

# program = clenergy_atomics_nways

# source = clenergy_atomics_nways.cu WKFUtils.cu

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

%.o: %.cu Makefile
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)

run_no_build:
	./$(program)

run_nsys:
#nsys profile -o H100_split_25ways_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)
#nsys profile -o A100_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)
	nsys profile -o RTX6000_vanila_high_res --gpu-metrics-device=all --gpu-metrics-frequency=50000 --cuda-graph-trace=node ./$(program)

run_ncu:
#ncu -o H100_ncu_report_vanila --set full ./$(program)
#ncu -o RTX6000_ncu_report_vanila --set full ./$(program)
	ncu -o A100_ncu_report_vanila --set full ./$(program)
