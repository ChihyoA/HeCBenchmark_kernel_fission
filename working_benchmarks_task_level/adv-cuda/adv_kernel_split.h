__global__ void kernel_U1(
   const int Nelements,
   const dfloat *U,
   const dfloat *cubInterpT,
   const int offset,
   dfloat *U1_out
) {
   __shared__ dfloat s_cubInterpT[8][16];
   __shared__ dfloat s_U[8][8];
   __shared__ dfloat s_U1[16][16];

   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;
   const int id = j * 16 + i;

   if (j < 8 && i < 16) s_cubInterpT[j][i] = cubInterpT[id];

   dfloat r_U[16] = {0};

   for (int c = 0; c < 8; ++c) {
       if (j < 8 && i < 8) {
           const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
           s_U[j][i] = U[id + 0 * offset];
       }
       __syncthreads();

       if (j < 8) {
           dfloat U1 = 0;
           for (int a = 0; a < 8; ++a) {
               dfloat Iia = s_cubInterpT[a][i];
               U1 += Iia * s_U[j][a];
           }
           s_U1[j][i] = U1;
       } else {
           s_U1[j][i] = 0;
       }
       __syncthreads();

       dfloat U2 = 0;
       for (int b = 0; b < 8; ++b) {
           dfloat Ijb = s_cubInterpT[b][j];
           U2 += Ijb * s_U1[b][i];
       }
       
       for (int k = 0; k < 16; ++k) {
           dfloat Ikc = s_cubInterpT[c][k];
           r_U[k] += Ikc * U2;
       }
   }

   // Store results
   for (int k = 0; k < 16; ++k) {
       const int outId = e * p_cubNp + k * 16 * 16 + j * 16 + i;
       U1_out[outId] = r_U[k];
   }
}

__global__ void kernel_V1(
   const int Nelements,
   const dfloat *U,
   const dfloat *cubInterpT,
   const int offset,
   dfloat *V1_out
) {
   __shared__ dfloat s_cubInterpT[8][16];
   __shared__ dfloat s_V[8][8];
   __shared__ dfloat s_V1[16][16];

   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;
   const int id = j * 16 + i;

   if (j < 8 && i < 16) s_cubInterpT[j][i] = cubInterpT[id];

   dfloat r_V[16] = {0};

   for (int c = 0; c < 8; ++c) {
       if (j < 8 && i < 8) {
           const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
           s_V[j][i] = U[id + 1 * offset];
       }
       __syncthreads();

       if (j < 8) {
           dfloat V1 = 0;
           for (int a = 0; a < 8; ++a) {
               dfloat Iia = s_cubInterpT[a][i];
               V1 += Iia * s_V[j][a];
           }
           s_V1[j][i] = V1;
       } else {
           s_V1[j][i] = 0;
       }
       __syncthreads();

       dfloat V2 = 0;
       for (int b = 0; b < 8; ++b) {
           dfloat Ijb = s_cubInterpT[b][j];
           V2 += Ijb * s_V1[b][i];
       }

       for (int k = 0; k < 16; ++k) {
           dfloat Ikc = s_cubInterpT[c][k];
           r_V[k] += Ikc * V2;
       }
   }

   for (int k = 0; k < 16; ++k) {
       const int outId = e * p_cubNp + k * 16 * 16 + j * 16 + i;
       V1_out[outId] = r_V[k];
   }
}

__global__ void kernel_W1(
   const int Nelements,
   const dfloat *U,
   const dfloat *cubInterpT,
   const int offset,
   dfloat *W1_out
) {
   __shared__ dfloat s_cubInterpT[8][16];
   __shared__ dfloat s_W[8][8];
   __shared__ dfloat s_W1[16][16];

   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;
   const int id = j * 16 + i;

   if (j < 8 && i < 16) s_cubInterpT[j][i] = cubInterpT[id];

   dfloat r_W[16] = {0};

   for (int c = 0; c < 8; ++c) {
       if (j < 8 && i < 8) {
           const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
           s_W[j][i] = U[id + 2 * offset];
       }
       __syncthreads();

       if (j < 8) {
           dfloat W1 = 0;
           for (int a = 0; a < 8; ++a) {
               dfloat Iia = s_cubInterpT[a][i];
               W1 += Iia * s_W[j][a];
           }
           s_W1[j][i] = W1;
       } else {
           s_W1[j][i] = 0;
       }
       __syncthreads();

       dfloat W2 = 0;
       for (int b = 0; b < 8; ++b) {
           dfloat Ijb = s_cubInterpT[b][j];
           W2 += Ijb * s_W1[b][i];
       }

       for (int k = 0; k < 16; ++k) {
           dfloat Ikc = s_cubInterpT[c][k];
           r_W[k] += Ikc * W2;
       }
   }

   for (int k = 0; k < 16; ++k) {
       const int outId = e * p_cubNp + k * 16 * 16 + j * 16 + i;
       W1_out[outId] = r_W[k];
   }
}

__global__ void kernel_U_final(
   const int Nelements,
   const dfloat *U1_in,
   const dfloat *cubD,
   const dfloat *cubvgeo,
   const int offset,
   dfloat *NU
) {
   __shared__ dfloat s_cubD[16][16];
   __shared__ dfloat s_U1[16][16];
   
   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;

   s_cubD[j][i] = cubD[j * 16 + i];
   
   for (int k = 0; k < 16; ++k) {
       s_U1[j][i] = U1_in[e * p_cubNp + k * 16 * 16 + j * 16 + i];
       __syncthreads();

       dfloat Udr = 0;
       for (int n = 0; n < 16; ++n) {
           dfloat Din = s_cubD[i][n];
           Udr += Din * s_U1[j][n];
       }

       const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
       const dfloat drdx = cubvgeo[gid + p_RXID * p_cubNp];
       const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];

       if (i < 8 && j < 8) {
           const int id = e * p_Np + k * 8 * 8 + j * 8 + i;
           NU[id + 0 * offset] = JW * drdx * Udr;
       }
   }
}

__global__ void kernel_V_final(
   const int Nelements,
   const dfloat *V1_in,
   const dfloat *cubD,
   const dfloat *cubvgeo,
   const int offset,
   dfloat *NU
) {
   __shared__ dfloat s_cubD[16][16];
   __shared__ dfloat s_V1[16][16];
   
   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;

   s_cubD[j][i] = cubD[j * 16 + i];
   
   for (int k = 0; k < 16; ++k) {
       s_V1[j][i] = V1_in[e * p_cubNp + k * 16 * 16 + j * 16 + i];
       __syncthreads();

       dfloat Vdr = 0;
       for (int n = 0; n < 16; ++n) {
           dfloat Din = s_cubD[i][n];
           Vdr += Din * s_V1[j][n];
       }

       const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
       const dfloat drdy = cubvgeo[gid + p_RYID * p_cubNp];
       const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];

       if (i < 8 && j < 8) {
           const int id = e * p_Np + k * 8 * 8 + j * 8 + i;
           NU[id + 1 * offset] = JW * drdy * Vdr;
       }
   }
}

__global__ void kernel_W_final(
   const int Nelements,
   const dfloat *W1_in,
   const dfloat *cubD,
   const dfloat *cubvgeo,
   const int offset,
   dfloat *NU
) {
   __shared__ dfloat s_cubD[16][16];
   __shared__ dfloat s_W1[16][16];
   
   const int e = blockIdx.x;
   const int i = threadIdx.x;
   const int j = threadIdx.y;

   s_cubD[j][i] = cubD[j * 16 + i];
   
   for (int k = 0; k < 16; ++k) {
       s_W1[j][i] = W1_in[e * p_cubNp + k * 16 * 16 + j * 16 + i];
       __syncthreads();

       dfloat Wdr = 0;
       for (int n = 0; n < 16; ++n) {
           dfloat Din = s_cubD[i][n];
           Wdr += Din * s_W1[j][n];
       }

       const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
       const dfloat drdz = cubvgeo[gid + p_RZID * p_cubNp];
       const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];

       if (i < 8 && j < 8) {
           const int id = e * p_Np + k * 8 * 8 + j * 8 + i;
           NU[id + 2 * offset] = JW * drdz * Wdr;
       }
   }
}