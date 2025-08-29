#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "mv.h"

// sparse matrix vector multiply using the CSR format
__global__ void mv_csr_split_atomic(const int num_rows,
                       const size_t *row_indices,
                       const int *col_indices,
                       const REAL *values,
                       const REAL *x,
                             REAL *y,
                       const int stream_A,
                       const int num_streams)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    size_t row_start = row_indices[i];
    size_t row_end = row_indices[i+1];

    if (stream_A == 0)
    {
      y[i] = 0;
    }

    REAL temp = 0;

      // 나눠진 index 범위 정의
    const size_t split_start = stream_A * (row_end - row_start) / num_streams + row_start;
    const size_t split_end   = stream_A + 1 == num_streams ? row_end : (stream_A + 1) * (row_end - row_start) / num_streams + row_start;

    for(size_t n = split_start; n < split_end; n++){
      temp += values[n] * x[col_indices[n]];
    }

    atomicAdd(&y[i], temp);
    /*
    if (stream_A) {
      y[i] = temp;
    } else {
      y[i] += temp;
    }
    */

  }
}

// dense matrix vector multiply
__global__ void mv_dense(const int num_rows, const REAL* matrix, const REAL* x, REAL* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    REAL temp = 0;
    for (int j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != (REAL)0) 
        temp += matrix[i * num_rows + j] * x[j];
    }
    y[i] = temp;
  }
}

long mv_dense_parallel(const int repeat,
                       const int bs,
                       const int num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  REAL *d_x, *d_matrix, *d_y;
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
MY_START_CLOCK(cuda simpleSpmv-cuda kernels.cu,0);

  for (int i = 0; i < repeat; i++)
    mv_dense<<<grids, blocks>>>(num_rows, d_matrix, d_x, d_y);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
MY_STOP_CLOCK(cuda simpleSpmv-cuda kernels.cu,0);
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_matrix);

  return time;
}

long mv_csr_parallel(const int repeat,
                     const int bs,
                     const int num_rows,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  size_t *row_indices = (size_t *) malloc((num_rows+1) * sizeof(size_t));
  int *col_indices = (int *) malloc(nnz * sizeof(int));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

  size_t *d_row_indices;
  int *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  cudaMalloc(&d_col_indices, nnz*sizeof(int));
  cudaMalloc(&d_values, nnz*sizeof(REAL));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  //before the computation

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  MY_START_CLOCK(cuda simpleSpmv-cuda kernels.cu,1);

  // N-way split define
  int n_way_split = 10;
  cudaStream_t* streams = new cudaStream_t[n_way_split];
  for (int i = 0; i < n_way_split; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  for (int i = 0; i < repeat; i++)
  {
    cudaMemset(d_y, 0, num_rows*sizeof(REAL)); // Initialize y to zero 

    for (int j = 0; j < n_way_split; ++j) {
      mv_csr_split_atomic<<<grids, blocks, 0, streams[j]>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y, j, n_way_split); // N-way split
      }

      for (int j = 0; j < n_way_split; ++j) {
        cudaStreamSynchronize(streams[j]);
      }
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
MY_STOP_CLOCK(cuda simpleSpmv-cuda kernels.cu,1);

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  free(values);
  free(row_indices);
  free(col_indices);

  cudaFree(d_row_indices);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return time;
}
