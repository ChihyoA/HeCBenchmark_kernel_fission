#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cupti.h>

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 256

#define CUPTI_CALL(call)                                                        \
    do {                                                                        \
        CUptiResult _status = call;                                            \
        if (_status != CUPTI_SUCCESS) {                                        \
            const char *errstr;                                                \
            cuptiGetResultString(_status, &errstr);                            \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s\n",\
                    __FILE__, __LINE__, #call, errstr);                        \
            exit(-1);                                                          \
        }                                                                       \
    } while (0)

__device__ half2 half_max(const half2 a, const half2 b) {
    const half2 sub = __hsub2(a, b);
    const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
    const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
    const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a),
                                         *reinterpret_cast<const unsigned*>(&b), sw);
    return *reinterpret_cast<const half2*>(&res);
}

template <typename T>
__global__
void hmax(T const *__restrict__ const a,
          T const *__restrict__ const b,
          T *__restrict__ const r,
          const size_t size)
{
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x;
         i < size; i += blockDim.x * gridDim.x)
        r[i] = half_max(a[i], b[i]);
}

void generateInput(half2 * a, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        float x = static_cast<float>(rand() % 1001);
        float y = static_cast<float>(rand() % 1001);
        a[i] = __floats2half2_rn(x, y);
    }
}

void profile_kernel_with_cupti(CUcontext context, CUdevice device)
{
    const char* metricNames[] = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    };

    CUpti_MetricID metricIds[2];
    for (int i = 0; i < 2; ++i) {
        CUPTI_CALL(cuptiMetricGetIdFromName(device, metricNames[i], &metricIds[i]));
    }

    CUpti_EventGroupSets* metricSets;
    CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricIds), metricIds, &metricSets));

    printf("\n[CUPTI Metrics Summary]\n");
    for (uint32_t s = 0; s < metricSets->numSets; s++) {
        CUpti_EventGroupSet& groupSet = metricSets->sets[s];
        // Enable all event groups
        for (uint32_t i = 0; i < groupSet.numEventGroups; i++)
            CUPTI_CALL(cuptiEventGroupEnable(groupSet.eventGroups[i]));

        // === Run kernel here ===
        printf("Launching kernel...\n");
        size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;
        half2 *d_a, *d_b, *d_r;
        cudaMalloc((void**)&d_a, size * sizeof(half2));
        cudaMalloc((void**)&d_b, size * sizeof(half2));
        cudaMalloc((void**)&d_r, size * sizeof(half2));
        hmax<half2><<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
        cudaDeviceSynchronize();
        // ========================

        for (int m = 0; m < 2; ++m) {
            CUpti_MetricValue metricValue;
            CUPTI_CALL(cuptiMetricGetValue(device, metricIds[m],
                                           groupSet.eventGroups, groupSet.numEventGroups,
                                           0 /* timeDuration */, &metricValue));
            double val;
            CUPTI_CALL(cuptiMetricValueGetDouble(metricValue, &val));
            printf("%s = %.2f %%\n", metricNames[m], val);
        }

        for (uint32_t i = 0; i < groupSet.numEventGroups; i++)
            CUPTI_CALL(cuptiEventGroupDisable(groupSet.eventGroups[i]));

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    }

    CUPTI_CALL(cuptiEventGroupSetsDestroy(metricSets));
}

int main(int argc, char *argv[])
{
    CUdevice device;
    CUcontext context;

    // CUDA 초기화
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // CUPTI로 kernel metric 수집
    profile_kernel_with_cupti(context, device);

    return 0;
}