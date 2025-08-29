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

# program = main

# source = main.cu FDTD3dReference.cu FDTD3dGPU.cu shrUtils.cu cmd_arg_reader.cu

# program = main_pipeline

# source = main_pipeline.cu FDTD3dReference.cu FDTD3dGPU_pipeline.cu shrUtils.cu cmd_arg_reader.cu

program = main_pipeline_sync

source = main_pipeline_sync.cu FDTD3dReference.cu FDTD3dGPU_pipeline_sync.cu shrUtils.cu cmd_arg_reader.cu

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

%.o: %.cu shrUtils.h cmd_arg_reader.h exception.h FDTD3dGPU.h
	$(NVCC) $(HEC_NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) --dimx=192 --dimy=184 --timesteps=90

run_no_build:
	./$(program) --dimx=192 --dimy=184 --timesteps=90

run_nsys:
	nsys profile -o H100_pipeline_sync_high --gpu-metrics-device=all --gpu-metrics-frequency=40000 --cuda-graph-trace=node ./$(program) --dimx=192 --dimy=184 --timesteps=90

run_ncu:
	ncu -o H100_ncu_report_vanila --kernel-name "finite_difference"  --set full ./$(program) ./$(program) --dimx=192 --dimy=184 --timesteps=2
