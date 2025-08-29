/**
 * This file is the modified read-only mixbench GPU micro-benchmark suite.
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

__global__ void benchmark_func_split(float *g_data, const int blockdim,
                               const int compute_iterations, const int stream_A, const int num_streams)
{
  const unsigned int blockSize = blockdim;
  const int stride = blockSize;
  int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
  const int big_stride = gridDim.x*blockSize*granularity; // 32*256*8 = 65536

  float tmps[granularity];

  // 나눠진 index 범위 정의
  const int split_start = stream_A * (fusion_degree / num_streams);
  const int split_end = stream_A + 1 == num_streams ? fusion_degree : (stream_A + 1) * (fusion_degree / num_streams);


  for(int k=split_start; k<split_end; k++) {
    #pragma unroll
    for(int j=0; j<granularity; j++) {
      // Load elements (memory intensive part)
      tmps[j] = g_data[idx+j*stride+k*big_stride];

      // Perform computations (compute intensive part)
      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+seed;
    }

    // Multiply add reduction
    float sum = 0.f;
    #pragma unroll
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];

    #pragma unroll
    for(int j=0; j<granularity; j++)
      g_data[idx+k*big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  float *d_cd;
  cudaMalloc((void**)&d_cd, size*sizeof(float));
  cudaMemcpy(d_cd, cd,  size*sizeof(float), cudaMemcpyHostToDevice);

  // N-way split define
  int n_way_split = 4;

  cudaStream_t* streams = new cudaStream_t[n_way_split];
  for (int i = 0; i < n_way_split; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  // warmup
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < n_way_split; ++j) {
      benchmark_func_split<<<grid_dim, block_dim, 0, streams[j]>>>(d_cd, block_dim, i, j, n_way_split);
    }
    
  }

  cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();MY_START_CLOCK(cuda mixbench-cuda main.cu,0);

  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < n_way_split; ++j) {
      benchmark_func_split<<<grid_dim, block_dim, 0, streams[j]>>>(d_cd, block_dim, i, j, n_way_split);
    }
    for (int j = 0; j < n_way_split; ++j) {
      cudaStreamSynchronize(streams[j]);
    }
    
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();MY_STOP_CLOCK(cuda mixbench-cuda main.cu,0);
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  cudaMemcpy(cd, d_cd, size*sizeof(float), cudaMemcpyDeviceToHost);

  // verification
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(cd);
  cudaFree(d_cd);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
