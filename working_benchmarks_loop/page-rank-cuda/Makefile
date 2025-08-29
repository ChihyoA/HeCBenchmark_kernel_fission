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

# program = page-rank_split_nways

# source = main_split_nways.cu

# 얘가 되는거
# program = page-rank_split_update_mem_access_interleaving

# source = main_split_update_mem_access_interleaving.cu

# program = page-rank_split_nways_update_mem_access_interleaving

# source = main_split_nways_update_mem_access_interleaving.cu

# program = page-rank_split

# source = main_split.cu

# program = page-rank_nways_interleaving

# source = main_split_nways_interleaving.cu

#program = page-rank_expand2

#source = main_expand2.cu

#program = page-rank_flat

#source = main_flat.cu

program = page-rank

source = main.cu

# program = page-rank_update_mem_access

# source = main_update_mem_access.cu

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
	./$(program) -n 20000 -i 100
run_no_build:
	./$(program) -n 20000 -i 100

run_nsys:
	nsys profile -o H100_vanila --gpu-metrics-device=all --gpu-metrics-frequency=20000 --cuda-graph-trace=node 	./$(program) -n 20000 -i 100

run_ncu_split:
	ncu -o H100_ncu_report_vanila  --set full ./$(program) -n 20000 -i 2

run_ncu:
	ncu -o H100_ncu_report_vanila --set full ./$(program) -n 20000 -i 2
