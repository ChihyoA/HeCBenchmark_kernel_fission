nvcc -arch=sm_90 motionsim_kernel-split-memory.cu -o particle-diffusion-kernel-split-memory
./particle-diffusion-kernel-split-memory 8000 100 21 10000