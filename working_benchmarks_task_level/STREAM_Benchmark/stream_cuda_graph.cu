/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)                 
    SCALE:      a(i) = q*b(i)               
    SUM:        a(i) = b(i) + c(i)          
    TRIAD:      a(i) = b(i) + q*c(i)        

  It measures the memory system on the device.
  The implementation is in double precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation

  Further modifications by: Ben Cumming, CSCS; Andreas Herten (JSC/FZJ)
*/

//#define NTIMES  20

#include <string>
#include <vector>

#include <iostream>

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

#include <sys/time.h>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "\n" \
                  << "Code: " << error << " Reason: " << cudaGetErrorString(error) << "\n"; \
        exit(1); \
    } \
} while(0)

typedef double real;

static double   avgtime[4] = {0}, maxtime[4] = {0},
        mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


void print_help()
{
    printf(
        "Usage: stream [-s] [-n <elements>] [-b <blocksize>]\n\n"
        "  -s\n"
        "        Print results in SI units (by default IEC units are used)\n\n"
        "  -n <elements>\n"
        "        Put <elements> values in the arrays\n"
        "        (defaults to 1<<26)\n\n"
        "  -b <blocksize>\n"
        "        Use <blocksize> as the number of threads in each block\n"
        "        (defaults to 192)\n"
    );
}

void parse_options(int argc, char** argv, bool& SI, int& N, int& blockSize, int& NTIMES)
{
    // Default values
    SI = false;
    //N = 1<<26;
    N = 100000;
    blockSize = 192;
    NTIMES = 20;

    int c;

    while ((c = getopt (argc, argv, "sn:t:b:h")) != -1)
        switch (c)
        {
            case 's':
                SI = true;
                break;
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'b':
                blockSize = std::atoi(optarg);
                break;
            case 'h':
                print_help();
                std::exit(0);
                break;
            case 't':
                NTIMES = std::atoi(optarg);
                break;
            default:
                print_help();
                std::exit(1);
        }
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


template <typename T>
__global__ void set_array(T * __restrict__ const a, T value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        a[idx] = value;
}

template <typename T>
__global__ void STREAM_Copy(T const * __restrict__ const a, T * __restrict__ const b, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        b[idx] = a[idx];
}

template <typename T>
__global__ void STREAM_Scale(T const * __restrict__ const a, T * __restrict__ const b, T scale,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        b[idx] = scale * a[idx];
}

template <typename T>
__global__ void STREAM_Add(T const * __restrict__ const a, T const * __restrict__ const b, T * __restrict__ const c, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx] + b[idx];
}

template <typename T>
__global__ void STREAM_Triad(T const * __restrict__ a, T const * __restrict__ b, T * __restrict__ const c, T scalar, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx] + scalar * b[idx];
}

template <typename T>
__global__ void stream_monolithic(T const * __restrict__ const a, T * __restrict__ const b,  T * __restrict__ const c, T scale,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < len) {
        // Copy
        b[idx] = a[idx];

        // Scale
        b[idx] = scale * a[idx];

        // Add
        c[idx] = a[idx] + b[idx];

        // Triad
        c[idx] = a[idx] + scale * b[idx];
    }
}

int main(int argc, char** argv)
{
    real *d_a, *d_b, *d_c;
    int j,k;
    real scalar;
    std::vector<std::string> label{"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

    // Parse arguments
    bool SI;
    int N, blockSize, NTIMES;
    parse_options(argc, argv, SI, N, blockSize, NTIMES);

    printf(" STREAM Benchmark implementation in CUDA\n");
    printf(" Array size (%s precision) =%7.2f MB\n", sizeof(double)==sizeof(real)?"double":"single", double(N)*double(sizeof(real))/1.e6);

    /* Allocate memory on device */
    cudaMalloc((void**)&d_a, sizeof(real)*N);
    cudaMalloc((void**)&d_b, sizeof(real)*N);
    cudaMalloc((void**)&d_c, sizeof(real)*N);

    /* Compute execution configuration */
    dim3 dimBlock(blockSize);
    dim3 dimGrid(N/dimBlock.x );
    if( N % dimBlock.x != 0 ) dimGrid.x+=1;

    printf(" using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    if (SI)
        printf(" output in SI units (KB = 1000 B)\n");
    else
        printf(" output in IEC units (KiB = 1024 B)\n");

    /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar=3.0f;

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaGraphCreate(&graph, 0);

    cudaStream_t stream0, stream1, stream2, stream3, graphStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreate(&graphStream));

    cudaEvent_t dummyEvent, event_stream_0, event_stream_1, event_stream_2, event_stream_3;
    cudaEventCreate(&dummyEvent);
    cudaEventCreate(&event_stream_0);
    cudaEventCreate(&event_stream_1);
    cudaEventCreate(&event_stream_2);
    cudaEventCreate(&event_stream_3);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    CUDA_CHECK(cudaStreamBeginCapture(graphStream, cudaStreamCaptureModeGlobal));

    /* Initialize memory on the device */
    set_array<real><<<dimGrid,dimBlock, 0, graphStream>>>(d_a, 2.f, N);
    set_array<real><<<dimGrid,dimBlock, 0, graphStream>>>(d_b, .5f, N);
    set_array<real><<<dimGrid,dimBlock, 0, graphStream>>>(d_c, .5f, N);
    CUDA_CHECK(cudaEventRecord(dummyEvent, graphStream));

    CUDA_CHECK(cudaStreamWaitEvent(stream0, dummyEvent, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream1, dummyEvent, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream2, dummyEvent, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream3, dummyEvent, 0));
    
    for (k=0; k<NTIMES; k++)
    {
        STREAM_Copy<real><<<dimGrid,dimBlock, 0, stream0>>>(d_a, d_c, N);
        CUDA_CHECK(cudaEventRecord(event_stream_0, stream0));

        STREAM_Scale<real><<<dimGrid,dimBlock, 0, stream1>>>(d_b, d_c, scalar,  N);
        CUDA_CHECK(cudaEventRecord(event_stream_1, stream1));

        STREAM_Add<real><<<dimGrid,dimBlock, 0, stream2>>>(d_a, d_b, d_c,  N);
        CUDA_CHECK(cudaEventRecord(event_stream_2, stream2));

        STREAM_Triad<real><<<dimGrid,dimBlock, 0, stream3>>>(d_b, d_c, d_a, scalar,  N);
        CUDA_CHECK(cudaEventRecord(event_stream_3, stream3));

        CUDA_CHECK(cudaStreamWaitEvent(graphStream, event_stream_0, 0));
        CUDA_CHECK(cudaStreamWaitEvent(graphStream, event_stream_1, 0));
        CUDA_CHECK(cudaStreamWaitEvent(graphStream, event_stream_2, 0));
        CUDA_CHECK(cudaStreamWaitEvent(graphStream, event_stream_3, 0));
    }

    CUDA_CHECK(cudaStreamEndCapture(graphStream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CUDA_CHECK(cudaGraphLaunch(graphExec, graphStream));
    CUDA_CHECK(cudaStreamSynchronize(graphStream));

    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Total Time: " << milliseconds << " ms" << std::endl;

    /* Free memory on device */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
