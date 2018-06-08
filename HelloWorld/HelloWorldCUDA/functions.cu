#include <stdio.h>
#include <cuda_runtime.h>
#include "functions.cuh"

__global__
void KernelAdd(int n, float * p_sum, float * x, float * y){

  for (int i = 0; i < n; i++)
    p_sum[i] = x[i] + y[i];
}

void DeviceKernel(int numBlocks, int sizeBlock, int n, float * p_sum, float * x, float * y){

    KernelAdd <<<numBlocks, sizeBlock>>> (n, p_sum, x, y);  // vector addition on DEVICE

}
