#include <stdio.h>
#include <cuda_runtime.h>

__global__
void KernelAdd(int n, float * p_sum, float * x, float * y){

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n){
    p_sum[i] = x[i] + y[i];

  return;
}
