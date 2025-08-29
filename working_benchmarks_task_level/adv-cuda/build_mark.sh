nvcc main_combined.cu -Ddfloat=double -Ddlong=int -std=c++14 -o adv-cuda-kernel-split-combined-mark

#./adv-cuda-kernel-split-combined-mark 7 15 128 100
./adv-cuda-kernel-split-combined-mark 7 15 8000 100