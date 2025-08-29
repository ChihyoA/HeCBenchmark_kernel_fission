__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

__device__ unsigned LCGStep(unsigned &z)
{
  return z = (1664525 * z + 1013904223);
}

// Uniform, need to do box muller on this
__device__ float getRandomValueTauswortheUniform(unsigned &z1, unsigned &z2, unsigned &z3, unsigned &z4)
{
  unsigned taus = TausStep(z1, 13, 19, 12, 4294967294U) ^ 
                  TausStep(z2, 2, 25, 4, 4294967288U) ^ TausStep(z3, 3, 11, 17, 4294967280U);
  unsigned lcg = LCGStep(z4);

  return 2.3283064365387e-10f * (taus ^ lcg);  // taus+
}

__device__ void boxMuller(float u1, float u2, float &uo1, float &uo2)
{
  float z1 = sqrtf(-2.0f * logf(u1));
  float s1 = sinf(2.0f * PI * u2);
  float s2 = cosf(2.0f * PI * u2);
  uo1 = z1 * s1;
  uo2 = z1 * s2;
}

__device__ float getRandomValueTausworthe(unsigned &z1, unsigned &z2, unsigned &z3, 
                                          unsigned &z4, float &temporary, unsigned phase)
{
  if (phase & 1)
  {
    // Return second value of pair
    return temporary;
  }
  else
  {
    float t1, t2, t3;
    // Phase is even, generate pair, return first of values, store second
    t1 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
    t2 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
    boxMuller(t1, t2, t3, temporary);
    return t3;
  }
}

__device__ float tausworthe_lookback_sim(
    unsigned T, float VOL_0, float EPS_0, 
    float A_0, float A_1, float A_2, float S_0,
    float MU, unsigned &z1, unsigned &z2,
    unsigned &z3, unsigned &z4, float* path)
{
  float temp_random_value;
  float vol = VOL_0, eps = EPS_0;
  float s = S_0;
  int base = threadIdx.x;

  for (unsigned t = 0; t < T; t++)
  {
    // store the current asset price
    path[base] = s;
    base += LOOKBACK_TAUSWORTHE_NUM_THREADS;

    // time-varying volatility in the GARCH model
    vol = sqrtf(A_0 + A_1 * vol * vol + A_2 * eps * eps);

    // size of next asset movement depends in part on the size of the most recent movement
    eps = getRandomValueTausworthe(z1, z2, z3, z4, temp_random_value, t) * vol;
    // s may become infinite
    eps = fmaxf(fminf(eps, 1.f), -1.f);

    // next price
    s *= expf(MU + eps);
  }

  // Look back at path to find payoff
  float sum = 0;
  for (unsigned t = 0; t < T; t++)
  {
    base -= LOOKBACK_TAUSWORTHE_NUM_THREADS;
    sum += fmaxf(path[base] - s, 0.f);
  }
  return sum;
}

__global__ void tausworthe_lookback(
    unsigned num_cycles,
    const unsigned int *__restrict__ seedValues,
    float *__restrict__ simulationResultsMean,
    float *__restrict__ simulationResultsVariance,
    const float *__restrict__ g_VOL_0,
    const float *__restrict__ g_EPS_0,
    const float *__restrict__ g_A_0,
    const float *__restrict__ g_A_1,
    const float *__restrict__ g_A_2,
    const float *__restrict__ g_S_0,
    const float *__restrict__ g_MU,
    const bool stream_A,
    unsigned *__restrict__ z1_tmp,
    unsigned *__restrict__ z2_tmp,
    unsigned *__restrict__ z3_tmp,
    unsigned *__restrict__ z4_tmp,
    float *__restrict__ mean_tmp,
    float *__restrict__ variance_tmp)
{
  __shared__ float path[LOOKBACK_TAUSWORTHE_NUM_THREADS*LOOKBACK_MAX_T];

  unsigned address = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned z1, z2, z3, z4; 
  float mean, variance;


  if(stream_A)
  {
    // Initialise tausworth with seeds
    z1 = seedValues[address];
    z2 = seedValues[address +     TAUSWORTHE_TOTAL_NUM_THREADS];
    z3 = seedValues[address + 2 * TAUSWORTHE_TOTAL_NUM_THREADS];
    z4 = seedValues[address + 3 * TAUSWORTHE_TOTAL_NUM_THREADS];
    mean = 0;
    variance = 0;
  }

  else
  {
    z1 = z1_tmp[address];
    z2 = z2_tmp[address];
    z3 = z3_tmp[address];
    z4 = z4_tmp[address];
    mean = mean_tmp[address];
    variance = variance_tmp[address];
  }

  float VOL_0, EPS_0, A_0, A_1, A_2, S_0, MU;
  VOL_0 = g_VOL_0[address];
  EPS_0 = g_EPS_0[address];
  A_0 = g_A_0[address];
  A_1 = g_A_1[address];
  A_2 = g_A_2[address];
  S_0 = g_S_0[address];
  MU = g_MU[address];

  
  
  // 나눠진 index 범위 정의
  const int split_start = stream_A ? 1 : (LOOKBACK_PATHS_PER_SIM / 2 - 1);
  const int split_end   = stream_A ? (LOOKBACK_PATHS_PER_SIM / 2) : LOOKBACK_PATHS_PER_SIM;

  for (unsigned i = split_start; i <= split_end; i++)
  {
    
    // simulate a path for num_cyles cyles 
    float res = tausworthe_lookback_sim(num_cycles, VOL_0, EPS_0,
        A_0, A_1, A_2, S_0,
        MU,
        z1, z2, z3, z4,  // rng state variables
        path);

    // update mean and variance in a numerically stable way
    float delta = res - mean;
    mean += delta / i;
    variance += delta * (res - mean);
  }

  if(stream_A)
  {
    // Store the updated seeds back to the global memory
    z1_tmp[address] = z1;
    z2_tmp[address] = z2;
    z3_tmp[address] = z3;
    z4_tmp[address] = z4;
    mean_tmp[address] = mean;
    variance_tmp[address] = variance;
  }
  else 
  {
    simulationResultsMean[address] = mean;
    simulationResultsVariance[address] = variance / (LOOKBACK_PATHS_PER_SIM - 1);
  }
}
