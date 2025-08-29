#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotate_matrix_parallel_pipeline (float *matrix, const int n, const bool stream_A) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;

    // 나눠진 index 범위 정의
    const int split_start = stream_A ? first : (((last-first) / 2 ) + first);
    const int split_end   = stream_A ? (((last-first) / 2 ) + first): last;
    
    //printf("split_start: %d, split_end: %d\n", split_start, split_end);

    for(int i = split_start; i < split_end; ++i) {
      int offset = i - first;

      float top = matrix[first*n+i]; // save top
      // left -> top
      matrix[first*n+i] = matrix[(last-offset)*n+first];

      // bottom -> left
      matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

      // right -> bottom
      matrix[last*n+(last-offset)] = matrix[i*n+last];

      // top -> right
      matrix[i*n+last] = top; // right <- saved top
    }
  }
}

void rotate_matrix_serial(float *matrix, int n) {

  for (int layer = 0; layer < n / 2; ++layer) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;
        float top = matrix[first*n+i]; // save top
        // left -> top
        matrix[first*n+i] = matrix[(last-offset)*n+first];

        // bottom -> left
        matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

        // right -> bottom
        matrix[last*n+(last-offset)] = matrix[i*n+last];

        // top -> right
        matrix[i*n+last] = top; // right <- saved top
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *serial_res = (float*) aligned_alloc(1024, n*n*sizeof(float));
  float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i*n+j] = parallel_res[i*n+j] = i*n+j;

  for (int i = 0; i < repeat; i++) {
    rotate_matrix_serial(serial_res, n);
  }

  float *d_parallel_res;
  cudaMalloc((void**)&d_parallel_res, n*n*sizeof(float));
  cudaMemcpy(d_parallel_res, parallel_res, n*n*sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);


  cudaEvent_t event;
  cudaEventCreate(&event);

  auto start=std::chrono::steady_clock::now();MY_START_CLOCK(cuda matrix-rotate-cuda main.cu,0);
  
  for (int i = 0; i < repeat; i++) {

    rotate_matrix_parallel_pipeline<<<(n/2+255)/256, 256, 0, stream1>>>(d_parallel_res, n, true);

    rotate_matrix_parallel_pipeline<<<(n/2+255)/256, 256, 0, stream2>>>(d_parallel_res, n, false);
    cudaDeviceSynchronize();
    
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();MY_STOP_CLOCK(cuda matrix-rotate-cuda main.cu,0);
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(parallel_res, d_parallel_res, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i*n+j] != parallel_res[i*n+j]) {
        ok = false;
        break;
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(serial_res);
  free(parallel_res);
  cudaFree(d_parallel_res);
  return 0;
}
