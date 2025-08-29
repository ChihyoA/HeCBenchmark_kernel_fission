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



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V,  //Volatility rate
    int iterations
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    
    // Add extra computations that cancel out exactly
    for(int i = 0; i < iterations; i++) {  // Can adjust iterations
        float temp = sqrtT;
        temp = temp * 1.5f;
        temp = temp / 1.5f;
        sqrtT = temp;  // Result stays exactly the same
    }
    
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
//First kernel writes directly to .x components
//__launch_bounds__(128)
__global__ void BlackScholesGPU_X(
    float2* __restrict d_CallResult,    // Write directly to final output
    float2* __restrict d_PutResult,     // Write directly to final output
    const float2* __restrict d_StockPrice,
    const float2* __restrict d_OptionStrike,
    const float2* __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int iterations
)
{
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
   
    if (opt < (optN / 2))
    {
        //if (opt < 4) { // Debug first few elements
        //    printf("X[%d]: KERNEL ENTRY: BlockIdx=%d, ThreadIdx=%d\n", 
        //           opt, blockIdx.x, threadIdx.x);
        //    printf("X[%d]: Input Stock=%.6f Strike=%.6f Years=%.6f\n",
        //           opt, d_StockPrice[opt].x, d_OptionStrike[opt].x, d_OptionYears[opt].x);
        //    printf("X[%d]: Addresses - Call=%p Put=%p\n", 
        //           opt, (void*)&d_CallResult[opt].x, (void*)&d_PutResult[opt].x);
	//}

	float callResult, putResult;
        BlackScholesBodyGPU(
            callResult,
            putResult,
            d_StockPrice[opt].x,
            d_OptionStrike[opt].x,
            d_OptionYears[opt].x,
            Riskfree,
            Volatility,
	    iterations
        );

        //if (opt < 4) {
        //    printf("X[%d]: Writing Call=%.6f Put=%.6f\n", opt, callResult, putResult);
        //}

        d_CallResult[opt].x = callResult;  // Write directly to .x
        d_PutResult[opt].x = putResult;    // Write directly to .x
    }
}

//Second kernel writes directly to .y components
//__launch_bounds__(128)
__global__ void BlackScholesGPU_Y(
    float2* __restrict d_CallResult,    // Write directly to final output
    float2* __restrict d_PutResult,     // Write directly to final output
    const float2* __restrict d_StockPrice,
    const float2* __restrict d_OptionStrike,
    const float2* __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int iterations
)
{
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (opt < (optN / 2))
    {
        //if (opt < 4) {
	//    printf("Y[%d]: KERNEL ENTRY: BlockIdx=%d, ThreadIdx=%d\n", 
        //           opt, blockIdx.x, threadIdx.x);
        //    printf("Y[%d]: Input Stock=%.6f Strike=%.6f Years=%.6f\n",
        //           opt, d_StockPrice[opt].y, d_OptionStrike[opt].y, d_OptionYears[opt].y);
        //    printf("Y[%d]: Addresses - Call=%p Put=%p\n", 
        //           opt, (void*)&d_CallResult[opt].y, (void*)&d_PutResult[opt].y);
	//}
	    
	float callResult, putResult;
        BlackScholesBodyGPU(
            callResult,
            putResult,
            d_StockPrice[opt].y,
            d_OptionStrike[opt].y,
            d_OptionYears[opt].y,
            Riskfree,
            Volatility,
	    iterations
        );

        //if (opt < 4) {
        //    printf("Y[%d]: Writing Call=%.6f Put=%.6f\n", opt, callResult, putResult);
        //}

        d_CallResult[opt].y = callResult;  // Write directly to .y
        d_PutResult[opt].y = putResult;    // Write directly to .y
    }
}



