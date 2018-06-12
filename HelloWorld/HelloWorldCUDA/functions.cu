#include <stdio.h>
#include <cuda_runtime.h>
#include "functions.cuh"

__global__
void KernelAdd(int n, float * p_sum, float * x, float * y){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < n )
    p_sum[i] = x[i] + y[i];
}
void DeviceKernel(int numBlocks, int sizeBlock, int n, float * p_sum, float * x, float * y){

    KernelAdd <<<numBlocks, sizeBlock>>> (n, p_sum, x, y);  // vector addition on DEVICE simple

}
