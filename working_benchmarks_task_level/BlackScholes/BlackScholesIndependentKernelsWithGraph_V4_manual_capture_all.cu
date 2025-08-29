/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include "helper_functions.h"   // helper functions for string parsing
#include "helper_cuda.h"        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel_split_independent.cuh"

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "\n" \
                  << "Code: " << error << " Reason: " << cudaGetErrorString(error) << "\n"; \
        exit(1); \
    } \
} while(0)

__global__ void dummyKernel(int x) {
    // Do nothing
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//const int OPT_N = 25000; // 4000000;
//const int  NUM_ITERATIONS = 1; //512;


//const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int OPT_N = std::atoi(argv[1]);
    int NUM_ITERATIONS = std::atoi(argv[2]);
    int threadsPerBlock = std::atoi(argv[3]);
    int iterations = std::atoi(argv[4]);

    const int OPT_SZ = OPT_N * sizeof(float);

    // Start logs
    if (argc != 5)
    {
       std::cerr << "Usage: " << argv[0] << "<OPT_N> <NUM_ITERATIONS (main for loop)> <threadsPerBlock> <iterations (kernel loop)>\n";
       return 1;
    }

    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    float *callResultX, *callResultY, *putResultX, *putResultY;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    size_t intermediate_size = (OPT_N/2) * sizeof(float);

    checkCudaErrors(cudaMalloc((void **)&callResultX, intermediate_size));
    checkCudaErrors(cudaMalloc((void **)&callResultY, intermediate_size));
    checkCudaErrors(cudaMalloc((void **)&putResultX, intermediate_size));
    checkCudaErrors(cudaMalloc((void **)&putResultY, intermediate_size));
    
    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");

    // Create graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0); // 
    cudaGraphExec_t graphExec;

    // Create streams with non-blocking flag

    const int NUM_STREAMS = 8;  // Adjust based on GPU capability and workload
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
    }
    //cudaStream_t stream1, stream2, graphStream;
    
    //cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    //cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    //cudaStreamCreate(&graphStream);

    // Create dummy event for root node
    //cudaEvent_t event_x, event_y;
    //cudaEventCreate(&dummyEvent);
    //cudaEventCreate(&event_x);
    //cudaEventCreate(&event_y);

    //printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    //checkCudaErrors(cudaDeviceSynchronize());
    //sdkResetTimer(&hTimer);
    //sdkStartTimer(&hTimer);
    
    printf("...copying input data to GPU mem.\n");
    float2 h_stock[4], h_strike[4], h_years[4];
    cudaMemcpy(h_stock, d_StockPrice, 4 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_strike, d_OptionStrike, 4 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_years, d_OptionYears, 4 * sizeof(float2), cudaMemcpyDeviceToHost);

    /*printf("Input Values for first 4 elements:\n");
    for(int i = 0; i < 4; i++) {
        printf("[%d] Stock(%.6f,%.6f) Strike(%.6f,%.6f) Years(%.6f,%.6f)\n",
        i, h_stock[i].x, h_stock[i].y, 
        h_strike[i].x, h_strike[i].y,
        h_years[i].x, h_years[i].y);
    }*/

    //const int threadsPerBlock = 128;

    // Create CUDA events for timing
    printf("Creating CUDA events for timing...\n");
    cudaEvent_t startGraph, stopGraph;
    cudaEvent_t startExec, stopExec;
    cudaEventCreate(&startGraph);
    cudaEventCreate(&stopGraph);
    cudaEventCreate(&startExec);
    cudaEventCreate(&stopExec);

    float graphTime = 0.0f;
    float execTime = 0.0f;

    
    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Time graph creation
    cudaEventRecord(startGraph);

    //cudaStreamBeginCapture(graphStream, cudaStreamCaptureModeGlobal);
    //cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    // Add dummy kernel as root node
    //dummyKernel<<<1, 1, 0, graphStream>>>(NUM_ITERATIONS);
    //cudaEventRecord(dummyEvent, graphStream);

    // Set up initial stream dependencies
    //cudaStreamWaitEvent(stream1, dummyEvent);
    //cudaStreamWaitEvent(stream2, dummyEvent);

    /*
    // Launch first kernel on stream1
    BlackScholesGPU_X<<<DIV_UP((OPT_N/2), threadsPerBlock), threadsPerBlock, 0, stream1>>>(
        (float2*)d_CallResult, (float2*)d_PutResult,
        (const float2*)d_StockPrice, 
        (const float2*)d_OptionStrike, 
        (const float2*)d_OptionYears,
        RISKFREE, VOLATILITY, OPT_N, iterations
    );
    cudaEventRecord(event_x, stream1);

    // Launch second kernel on stream2
    BlackScholesGPU_Y<<<DIV_UP((OPT_N/2), threadsPerBlock), threadsPerBlock, 0, stream2>>>(
        (float2*)d_CallResult, (float2*)d_PutResult,
        (const float2*)d_StockPrice, 
        (const float2*)d_OptionStrike, 
        (const float2*)d_OptionYears,
        RISKFREE, VOLATILITY, OPT_N, iterations
    );
    cudaEventRecord(event_y, stream2);

    */

    for (i = 0; i < NUM_ITERATIONS; i++) {

    cudaKernelNodeParams kernelNodeParams_X = {};
    void* kernelArgs_X[] = { &d_CallResult, &d_PutResult, &d_StockPrice, 
                            &d_OptionStrike, &d_OptionYears, 
                            (void*)&RISKFREE, (void*)&VOLATILITY, &OPT_N, &iterations };
    kernelNodeParams_X.func = (void*)BlackScholesGPU_X;
    kernelNodeParams_X.gridDim = dim3(DIV_UP((OPT_N/2), threadsPerBlock));
    kernelNodeParams_X.blockDim = dim3(threadsPerBlock);
    kernelNodeParams_X.sharedMemBytes = 0;
    kernelNodeParams_X.kernelParams = kernelArgs_X;
    kernelNodeParams_X.extra = nullptr;

    cudaGraphNode_t node_X;
    cudaGraphAddKernelNode(&node_X, graph, nullptr, 0, &kernelNodeParams_X);

    cudaKernelNodeParams kernelNodeParams_Y = {};
    void* kernelArgs_Y[] = { &d_CallResult, &d_PutResult, &d_StockPrice, 
                            &d_OptionStrike, &d_OptionYears, 
                            (void*)&RISKFREE, (void*)&VOLATILITY, &OPT_N, &iterations };
    kernelNodeParams_Y.func = (void*)BlackScholesGPU_Y;
    kernelNodeParams_Y.gridDim = dim3(DIV_UP((OPT_N/2), threadsPerBlock));
    kernelNodeParams_Y.blockDim = dim3(threadsPerBlock);
    kernelNodeParams_Y.sharedMemBytes = 0;
    kernelNodeParams_Y.kernelParams = kernelArgs_Y;
    kernelNodeParams_Y.extra = nullptr;

    cudaGraphNode_t node_Y;
    cudaGraphAddKernelNode(&node_Y, graph, nullptr, 0, &kernelNodeParams_Y);

    //cudaStreamWaitEvent(graphStream, event_x);
    //cudaStreamWaitEvent(graphStream, event_y);


    //cudaStreamEndCapture(stream1, &graph);
    //cudaStreamEndCapture(graphStream, &graph);

    }


    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaEventRecord(stopGraph);
    cudaEventSynchronize(stopGraph);
    cudaEventElapsedTime(&graphTime, startGraph, stopGraph);

    // Time graph execution
    cudaEventRecord(startExec);

        //cudaGraphLaunch(graphExec, graphStream);
    cudaGraphLaunch(graphExec, 0);
                // Use streams in a round-robin fashion
        //int streamIdx = i % NUM_STREAMS;

        // Launch the graph asynchronously on the selected stream
        //cudaGraphLaunch(graphExec, streams[streamIdx]);
        //getLastCudaError("BlackScholesGPU() execution failed\n");
       
 	
    //cudaStreamSynchronize(graphStream);

    // Synchronize all streams at the end
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(stopExec);
    cudaEventSynchronize(stopExec);
    cudaEventElapsedTime(&execTime, startExec, stopExec);
    
    printf("Graph creation time : %f msec\n", graphTime);
    printf("Graph execution time: %f msec\n", execTime);

    //checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer); /// NUM_ITERATIONS;

    float2 h_call[4], h_put[4];
    cudaDeviceSynchronize();  // Make sure kernels are done
    cudaMemcpy(h_call, d_CallResult, 4 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_put, d_PutResult, 4 * sizeof(float2), cudaMemcpyDeviceToHost);

    /*printf("\nOutput Values for first 4 elements:\n");
    for(int i = 0; i < 4; i++) {
        printf("[%d] Call(%.6f,%.6f) Put(%.6f,%.6f)\n",
        i, h_call[i].x, h_call[i].y, 
        h_put[i].x, h_put[i].y);
    }*/

    //checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&hTimer);
    //gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    //printf("Options count             : %i     \n", 2 * OPT_N);
    
    //printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    //printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));
    printf("BlackScholesGPU() time (Total) : %f msec\n", gpuTime);
    //printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
    //       (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

    //printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


    //printf("Checking the results...\n");
    //printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

    //printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    //printf("L1 norm: %E\n", L1norm);
    //printf("Max absolute error: %E\n\n", max_delta);

    //printf("Shutting down...\n");
    //printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    //printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    //printf("Shutdown done.\n");

    //printf("\n[BlackScholes] - Test Summary\n");

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    //printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}

