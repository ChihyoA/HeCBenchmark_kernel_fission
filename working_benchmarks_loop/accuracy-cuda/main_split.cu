#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#define GPU_NUM_THREADS 256

__global__
void accuracy_kernel_split(
    const int N,
    const int D,
    const int top_k,
    const float* Xdata,
    const int* labelData,
    int* accuracy,
    bool stream_A)
{
  typedef cub::BlockReduce<int, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int count = 0;

  // 나눠진 index 범위 정의
  const int split_start = stream_A ? 0 : (N / 2);
  const int split_end   = stream_A ? (N / 2) : N;

  for (int row = split_start + blockIdx.x; row < split_end; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
  if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { 
    atomicAdd(accuracy, count);
  }
}
 
int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int); 
  const size_t data_size_bytes = data_size * sizeof(float); 

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims; 

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

  int *d_label;
  cudaMalloc((void**)&d_label, label_size_bytes);
  cudaMemcpy(d_label, label, label_size_bytes, cudaMemcpyHostToDevice);

  float *d_data;
  cudaMalloc((void**)&d_data, data_size_bytes);
  cudaMemcpy(d_data, data, data_size_bytes, cudaMemcpyHostToDevice);

  int *d_count;
  cudaMalloc((void**)&d_count, sizeof(int));

  cudaDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);


  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  for (int ngrid = nrows / 16; ngrid <= nrows; ngrid *= 2) {

    dim3 grid (ngrid);
    printf("Grid size is %d\n", ngrid);

    auto start=std::chrono::steady_clock::now();MY_START_CLOCK(cuda accuracy-cuda main.cu,0);

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel_split<<<grid, block, 0, stream1>>>(nrows, ndims, top_k, d_data, d_label, d_count, true);
      accuracy_kernel_split<<<grid, block, 0, stream2>>>(nrows, ndims, top_k, d_data, d_label, d_count, false);
      cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();MY_STOP_CLOCK(cuda accuracy-cuda main.cu,0);
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    bool ok = (count == count_ref);
    printf("%s\n", ok ? "PASS" : "FAIL");
    // printf("Accuracy = %f\n", (float)count / nrows);
  }

  cudaFree(d_label);
  cudaFree(d_data);
  cudaFree(d_count);

  free(label);
  free(data);

  return 0;
}
