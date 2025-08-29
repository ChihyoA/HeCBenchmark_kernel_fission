// ************************************************
// original authors: Lee Howes and David B. Thomas
// ************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "loopback.h"
#include "kernels_pipeline.cu"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dump> <repeat>\n", argv[0]);
    return 1;
  }
  // display device results when enabled
  const int dump = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t loopback_size = sizeof(float) * LOOKBACK_NUM_PARAMETER_VALUES;
  const size_t seed_size = sizeof(unsigned int) * TAUSWORTHE_NUM_SEEDS;

  float *lookback_VOL_0 = (float *) malloc(loopback_size);
  float *lookback_A_0 = (float *) malloc(loopback_size);
  float *lookback_A_1 = (float *) malloc(loopback_size);
  float *lookback_A_2 = (float *) malloc(loopback_size);
  float *lookback_S_0 = (float *) malloc(loopback_size);
  float *lookback_EPS_0 = (float *) malloc(loopback_size);
  float *lookback_MU = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsMean = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsVariance = (float *) malloc(loopback_size);

  for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i++)
  {
    lookback_VOL_0[i] = Rand();
    lookback_A_0[i] = Rand();
    lookback_A_1[i] = Rand();
    lookback_A_2[i] = Rand();
    lookback_S_0[i] = Rand();
    lookback_EPS_0[i] = Rand();
    lookback_MU[i] = Rand();
  }

  unsigned int *tauswortheSeeds = (unsigned int *) malloc(seed_size);
  for (unsigned i = 0; i < TAUSWORTHE_NUM_SEEDS; i++)
    tauswortheSeeds[i] = (uint)rand() + 16;

  float *d_lookback_VOL_0, *d_lookback_A_0, *d_lookback_A_1,
        *d_lookback_A_2, *d_lookback_S_0, *d_lookback_EPS_0, *d_lookback_MU;
  float *d_lookbackSimulationResultsMean, *d_lookbackSimulationResultsVariance;
  unsigned int *d_tauswortheSeeds;

  cudaMalloc((void**) &d_tauswortheSeeds, seed_size);
  cudaMalloc((void**) &d_lookback_VOL_0, loopback_size);
  cudaMalloc((void**) &d_lookback_A_0, loopback_size);
  cudaMalloc((void**) &d_lookback_A_1, loopback_size);
  cudaMalloc((void**) &d_lookback_A_2, loopback_size);
  cudaMalloc((void**) &d_lookback_S_0, loopback_size);
  cudaMalloc((void**) &d_lookback_EPS_0, loopback_size);
  cudaMalloc((void**) &d_lookback_MU, loopback_size);
  cudaMalloc((void**) &d_lookbackSimulationResultsMean, loopback_size);
  cudaMalloc((void**) &d_lookbackSimulationResultsVariance, loopback_size);

  cudaMemcpy(d_tauswortheSeeds, tauswortheSeeds, seed_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_VOL_0, lookback_VOL_0, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_A_0, lookback_A_0, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_A_1, lookback_A_1, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_A_2, lookback_A_2, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_S_0, lookback_S_0, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_EPS_0, lookback_EPS_0, loopback_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lookback_MU, lookback_MU, loopback_size, cudaMemcpyHostToDevice);

  // Execute the Tausworthe version of the lookback option
  dim3 grid (LOOKBACK_TAUSWORTHE_NUM_BLOCKS, 1, 1);
  dim3 threads (LOOKBACK_TAUSWORTHE_NUM_THREADS, 1, 1);
  const unsigned num_cycles = LOOKBACK_MAX_T;

  cudaDeviceSynchronize();

  auto start=std::chrono::steady_clock::now();MY_START_CLOCK(cuda loopback-cuda main.cu,0);

  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  unsigned int *z1_tmp, *z2_tmp, *z3_tmp, *z4_tmp;
  float *mean_tmp, *variance_tmp;
  cudaMalloc((void**)&z1_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(unsigned int));
  cudaMalloc((void**)&z2_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(unsigned int));
  cudaMalloc((void**)&z3_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(unsigned int));
  cudaMalloc((void**)&z4_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(unsigned int));
  cudaMalloc((void**)&mean_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(float));
  cudaMalloc((void**)&variance_tmp, LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_TAUSWORTHE_NUM_BLOCKS * sizeof(float));

  cudaEvent_t event[(repeat)];  // NUM = number of (A,B) pairs
  for (int i = 0; i < (repeat); ++i) {
      cudaEventCreate(&event[i]);
  }

  for (int i = 0; i < repeat; i++) {
    tausworthe_lookback <<< grid, threads , 0, stream1>>> (
       num_cycles,
       d_tauswortheSeeds,
       d_lookbackSimulationResultsMean,
       d_lookbackSimulationResultsVariance,
       d_lookback_VOL_0,
       d_lookback_EPS_0,
       d_lookback_A_0,
       d_lookback_A_1,
       d_lookback_A_2,
       d_lookback_S_0,
       d_lookback_MU,
       true,
       z1_tmp,
       z2_tmp,
       z3_tmp,
       z4_tmp,
       mean_tmp,
       variance_tmp);


       cudaEventRecord(event[i], stream1);
       cudaStreamWaitEvent(stream2, event[i], 0);

       tausworthe_lookback <<< grid, threads , 0, stream2>>> (
        num_cycles,
        d_tauswortheSeeds,
        d_lookbackSimulationResultsMean,
        d_lookbackSimulationResultsVariance,
        d_lookback_VOL_0,
        d_lookback_EPS_0,
        d_lookback_A_0,
        d_lookback_A_1,
        d_lookback_A_2,
        d_lookback_S_0,
        d_lookback_MU,
        false,
        z1_tmp,
        z2_tmp,
        z3_tmp,
        z4_tmp,
        mean_tmp,
        variance_tmp);

        //cudaDeviceSynchronize();

  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();MY_STOP_CLOCK(cuda loopback-cuda main.cu,0);
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(lookbackSimulationResultsMean, d_lookbackSimulationResultsMean,
             loopback_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(lookbackSimulationResultsVariance, d_lookbackSimulationResultsVariance,
             loopback_size, cudaMemcpyDeviceToHost);

  if (dump) {
    for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i+=10000)
      printf("%d %.3f %.3f\n", i, lookbackSimulationResultsMean[i], 
                              lookbackSimulationResultsVariance[i]);
  }

  free(lookback_VOL_0);
  free(lookback_A_0);
  free(lookback_A_1);
  free(lookback_A_2);
  free(lookback_S_0);
  free(lookback_EPS_0);
  free(lookback_MU);
  free(lookbackSimulationResultsMean);
  free(lookbackSimulationResultsVariance);
  free(tauswortheSeeds);

  cudaFree(d_tauswortheSeeds);
  cudaFree(d_lookback_VOL_0);
  cudaFree(d_lookback_A_0);
  cudaFree(d_lookback_A_1);
  cudaFree(d_lookback_A_2);
  cudaFree(d_lookback_S_0);
  cudaFree(d_lookback_EPS_0);
  cudaFree(d_lookback_MU);
  cudaFree(d_lookbackSimulationResultsMean);
  cudaFree(d_lookbackSimulationResultsVariance);

  return 0;
}
