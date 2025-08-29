nvcc -arch=sm_90 motionsim.cu -o particle-diffusion
./particle-diffusion-kernel-split-memory 8000 100 21 10000