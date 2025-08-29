nvcc main.cu -Ddfloat=double -Ddlong=int -std=c++14 -o adv-cuda

#./adv-cuda 7 15 128 100
./adv-cuda 7 15 8000 100