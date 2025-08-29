#include <sys/types.h>
#include <chrono>
#include <cuda.h>
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees  */
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor  */
#define FACTOR_CHIP  0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

__global__ void
hotspot3d(
    const float*__restrict__ tIn, 
    const float*__restrict__ pIn, 
          float*__restrict__ tOut,
    const int numCols, 
    const int numRows, 
    const int layers,
    const float ce, 
    const float cw,
    const float cn, 
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap,
    const bool stream_A,
    float*__restrict__ temp1_tmp,
    float*__restrict__ temp2_tmp,
    float*__restrict__ temp3_tmp
    )
{
  float amb_temp = 80.0;

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * numCols;
  int xy = numCols * numRows;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == numCols-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - numCols;
  int S = (j == numRows-1)     ? c : c + numCols;

  float temp1, temp2, temp3;

  // 나눠진 index 범위 정의
  const int split_start = stream_A ? 1 : ((layers-1)/2+1);
  const int split_end   = stream_A ? ((layers-1)/2+1) : layers - 1;
  int global_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  if (stream_A)
  {
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }

  else
  {
    temp1 = temp1_tmp[global_id];
    temp2 = temp2_tmp[global_id];
    temp3 = temp3_tmp[global_id];
    c += xy*((layers-1)/2+1);
    W += xy*((layers-1)/2+1);
    E += xy*((layers-1)/2+1);
    N += xy*((layers-1)/2+1);
    S += xy*((layers-1)/2+1);
  }

  
  for (int k = split_start; k < split_end; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }

  if(stream_A)
  {
    temp1_tmp[global_id] = temp1;
    temp2_tmp[global_id] = temp2;
    temp3_tmp[global_id] = temp3;
  }
  else
  {
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp; 
  }

}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw = stepDivCap/ Rx;
  cn               = cs = stepDivCap/ Ry;
  ct               = cb = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  float *d_tIn, *d_pIn, *d_tOut;
  cudaMalloc((void**)&d_tIn, sizeof(float)*size);
  cudaMalloc((void**)&d_pIn, sizeof(float)*size);
  cudaMalloc((void**)&d_tOut, sizeof(float)*size);

  cudaMemcpy(d_tIn, tIn,  sizeof(float)*size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_pIn, pIn,  sizeof(float)*size, cudaMemcpyHostToDevice); 

  dim3 gridDim(numCols/WG_SIZE_X, numRows/WG_SIZE_Y);
  dim3 blockDim(WG_SIZE_X, WG_SIZE_Y);

  cudaDeviceSynchronize();
  auto kstart=std::chrono::steady_clock::now();
MY_START_CLOCK(cuda hotspot3D-cuda 3D.cu,0);

float *temp1_tmp, *temp2_tmp, *temp3_tmp;
cudaMalloc((void**)&temp1_tmp, sizeof(float) * numCols * numRows);
cudaMalloc((void**)&temp2_tmp, sizeof(float) * numCols * numRows);
cudaMalloc((void**)&temp3_tmp, sizeof(float) * numCols * numRows);

  // Create two streams for pipelining

cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

cudaEvent_t event[(iterations)];  // NUM = number of (A,B) pairs
for (int i = 0; i < (iterations); ++i) {
    cudaEventCreate(&event[i]);
}

  for(int j = 0; j < iterations; j++)
  {

  
    hotspot3d<<<gridDim, blockDim, 0, stream1>>>(
        d_tIn, d_pIn, d_tOut, numCols, numRows, layers,
        ce, cw, cn, cs, ct, cb, cc, stepDivCap, true, temp1_tmp, temp2_tmp, temp3_tmp);

        cudaEventRecord(event[j], stream1);
        cudaStreamWaitEvent(stream2, event[j], 0);

        hotspot3d<<<gridDim, blockDim, 0, stream2>>>(
          d_tIn, d_pIn, d_tOut, numCols, numRows, layers,
          ce, cw, cn, cs, ct, cb, cc, stepDivCap, false, temp1_tmp, temp2_tmp, temp3_tmp);

          //cudaDeviceSynchronize();

    float* temp = d_tIn;
    d_tIn = d_tOut;
    d_tOut = temp;
  }

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono:: duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
MY_STOP_CLOCK(cuda hotspot3D-cuda 3D.cu,0);
  printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

  float* d_sel = (iterations & 01) ? d_tIn : d_tOut;
  cudaMemcpy(tOut, d_sel,  sizeof(float)*size, cudaMemcpyDeviceToHost); 
  cudaFree(d_tIn);
  cudaFree(d_pIn);
  cudaFree(d_tOut);
  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}
