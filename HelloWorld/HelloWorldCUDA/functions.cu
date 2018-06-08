#include "functions.cuh"

// cuda library
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

__global__
void KernelAdd(int n, float * p_sum, float * x, float * y){

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n){
    p_sum[i] = x[i] + y[i];

  return;
}
