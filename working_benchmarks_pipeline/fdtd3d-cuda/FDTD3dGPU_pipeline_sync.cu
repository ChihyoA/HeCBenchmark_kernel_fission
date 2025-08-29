/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include "FDTD3dGPU.h"
#include "shrUtils.h"

__global__ void finite_difference(
        float*__restrict__ output,
  const float*__restrict__ input,
  const float*__restrict__ coef, 
  const int dimx, const int dimy, const int dimz,
  const int padding,
  bool stream_A,
  float*__restrict__ infront_tmp,
  float*__restrict__ behind_tmp,
  float*__restrict__ current_tmp,
  int* __restrict__ inputIndex_tmp,
  int* __restrict__ outputIndex_tmp
  )
{
  bool valid = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;

  const int stride_y = dimx + 2 * k_radius_default;
  const int stride_z = stride_y * (dimy + 2 * k_radius_default);

  __shared__ float tile[k_blockDimMaxY + 2 * k_radius_default][k_blockDimMaxX + 2 * k_radius_default];

  int inputIndex  = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += k_radius_default * stride_y + k_radius_default + padding;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[k_radius_default];
  float behind[k_radius_default];
  float current;

  const int tx = ltidx + k_radius_default;
  const int ty = ltidy + k_radius_default;

  if (gtidx >= dimx) valid = false;
  if (gtidy >= dimy) valid = false;

  // For simplicity we assume that the global size is equal to the actual
  // problem size; since the global size must be a multiple of the local size
  // this means the problem size must be a multiple of the local size (or
  // padded to meet this constraint).
  // Preload the "infront" and "behind" data


  // 나눠진 index 범위 정의
  const int split_start = stream_A ? 0 : (dimz / 2);
  const int split_end   = stream_A ? (dimz / 2) : dimz;
  //int global_id = gtidy * (gridDim.x * workx) + gtidx;
  int global_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

 
  if (stream_A)
  {
    for (int i = k_radius_default - 2 ; i >= 0 ; i--)
    {
      behind[i] = input[inputIndex];
      inputIndex += stride_z;
    }
    current = input[inputIndex];
    outputIndex = inputIndex;
    inputIndex += stride_z;
  
    for (int i = 0 ; i < k_radius_default ; i++)
    {
      infront[i] = input[inputIndex];
      inputIndex += stride_z;
    }
  }
  else
  {
    for (int i = 0 ; i < k_radius_default; i++)
    {
      behind[i] = behind_tmp[global_id * k_radius_default + i];
      infront[i] = infront_tmp[global_id * k_radius_default + i];
    }
    current = current_tmp[global_id];
    outputIndex = outputIndex_tmp[global_id];
    inputIndex = inputIndex_tmp[global_id];
  }

  // Step through the xy-planes
  for (int iz = split_start ; iz < split_end ; iz++)
  {
    // Advance the slice (move the thread-front)
    for (int i = k_radius_default - 1 ; i > 0 ; i--)
      behind[i] = behind[i - 1];
      
    behind[0] = current;
    current = infront[0];
    for (int i = 0 ; i < k_radius_default - 1 ; i++)
      infront[i] = infront[i + 1];
    infront[k_radius_default - 1] = input[inputIndex];

    inputIndex  += stride_z;
    outputIndex += stride_z;
    __syncthreads();

      // Note that for the work items on the boundary of the problem, the
      // supplied index when reading the halo (below) may wrap to the
      // previous/next row or even the previous/next xy-plane. This is
      // acceptable since a) we disable the output write for these work
      // items and b) there is at least one xy-plane before/after the
      // current plane, so the access will be within bounds.

      // Update the data slice in the local tile
      // Halo above & below
      if (ltidy < k_radius_default)
      {
        tile[ltidy][tx]                  = input[outputIndex - k_radius_default * stride_y];
        tile[ltidy + worky + k_radius_default][tx] = input[outputIndex + worky * stride_y];
      }
    // Halo left & right
    if (ltidx < k_radius_default)
    {
      tile[ty][ltidx]                  = input[outputIndex - k_radius_default];
      tile[ty][ltidx + workx + k_radius_default] = input[outputIndex + workx];
    }
    tile[ty][tx] = current;
    __syncthreads();

      // Compute the output value
      float value = coef[0] * current;
    for (int i = 1 ; i <= k_radius_default ; i++)
    {
      value += coef[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + 
          tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    // Store the output value
    if (valid) output[outputIndex] = value;
  }

  if (stream_A)
  {
    // Store the "infront" and "behind" data for the next stream
    for (int i = 0 ; i < k_radius_default ; i++)
    {
      infront_tmp[global_id * k_radius_default + i] = infront[i];
      behind_tmp[global_id * k_radius_default + i] = behind[i];
    }
    inputIndex_tmp[global_id] = inputIndex;
    outputIndex_tmp[global_id] = outputIndex;
    current_tmp[global_id] = current;
  }

}

bool fdtdGPU(float *output, const float *input, const float *coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char **argv)
{
  bool ok = true;
  const int         outerDimx  = dimx + 2 * radius;
  const int         outerDimy  = dimy + 2 * radius;
  const int         outerDimz  = dimz + 2 * radius;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
  size_t            gridSize[2];
  size_t            blockSize[2];

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;


  // Create memory buffer objects
  float* bufferOut; 
  cudaMalloc((void**)&bufferOut, paddedVolumeSize * sizeof(float));

  float* bufferIn; 
  cudaMalloc((void**)&bufferIn, paddedVolumeSize * sizeof(float));

  float* bufferCoef; 
  cudaMalloc((void**)&bufferCoef, (radius+1) * sizeof(float));
  cudaMemcpy(bufferCoef, coeff, (radius+1) * sizeof(float), cudaMemcpyHostToDevice);

  // Set the maximum work group size
  size_t maxWorkSize = 256;

  // Set the work group size
  blockSize[0] = k_localWorkX;  
  blockSize[1] = maxWorkSize / k_localWorkX;
  gridSize[0] = (unsigned int)ceil((float)dimx / blockSize[0]);
  gridSize[1] = (unsigned int)ceil((float)dimy / blockSize[1]);
  shrLog(" set block size to %dx%d\n", blockSize[0], blockSize[1]);
  shrLog(" set grid size to %dx%d\n", gridSize[0], gridSize[1]);
  dim3 grid (gridSize[0], gridSize[1]);
  dim3 block (blockSize[0], blockSize[1]);

  // Copy the input to the device input buffer 
  // offset = padding * 4, bytes = volumeSize * 4
  cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice);

  // Copy the input to the device output buffer (actually only need the halo)
  cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice);

  auto start=std::chrono::steady_clock::now();MY_START_CLOCK(cuda fdtd3d-cuda FDTD3dGPU.cu,0);

  float *infront_tmp, *behind_tmp, *current_tmp;
  int *inputIndex_tmp, *outputIndex_tmp;
  cudaMalloc((void**)&infront_tmp, k_radius_default * gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1] * sizeof(float));
  cudaMalloc((void**)&behind_tmp, k_radius_default * gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1] * sizeof(float));
  cudaMalloc((void**)&current_tmp, gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1] * sizeof(float));
  cudaMalloc((void**)&inputIndex_tmp, gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1] * sizeof(int));
  cudaMalloc((void**)&outputIndex_tmp, gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1] * sizeof(int));

  // Execute the FDTD
  shrLog(" GPU FDTD loop\n");

  cudaDeviceSynchronize();

  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  cudaEvent_t event[(timesteps)], event2[(timesteps)];  // NUM = number of (A,B) pairs
  for (int i = 0; i < (timesteps); ++i) {
      cudaEventCreate(&event[i]);
      cudaEventCreate(&event2[i]);
  }

  

  for (int it = 0 ; it < timesteps ; it++)
  {
    // Launch the kernel
    
    size_t total_threads = gridSize[0] * gridSize[1] * blockSize[0] * blockSize[1];

    if (it >= 2)
    {
      cudaStreamWaitEvent(stream1, event2[it - 2], 0);
    }

    finite_difference<<<grid, block, 0, stream1>>>(bufferOut, bufferIn, bufferCoef, dimx, dimy, dimz, padding, true, infront_tmp, behind_tmp, current_tmp, inputIndex_tmp, outputIndex_tmp);

    cudaEventRecord(event[it], stream1);
    cudaStreamWaitEvent(stream2, event[it], 0);

    finite_difference<<<grid, block, 0, stream2>>>(bufferOut, bufferIn, bufferCoef, dimx, dimy, dimz, padding, false, infront_tmp, behind_tmp, current_tmp, inputIndex_tmp, outputIndex_tmp);

    cudaEventRecord(event2[it], stream2);
    // Toggle the buffers
    float* tmp = bufferIn;
    bufferIn = bufferOut;
    bufferOut = tmp;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();MY_STOP_CLOCK(cuda fdtd3d-cuda FDTD3dGPU.cu,0);
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);

  // Read the result back, result is in bufferSrc (after final toggle)
  cudaMemcpy(output, bufferIn + padding, volumeSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(bufferIn);
  cudaFree(bufferOut);
  cudaFree(bufferCoef);
  return ok;
}
