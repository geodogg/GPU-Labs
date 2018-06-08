#include <stdio.h>
#include <cuda_runtime.h>
#include <functions.cuh>

__global__
void KernelAdd(int n, float * p_sum, float * x, float * y){

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n){
    p_sum[i] = x[i] + y[i];

  return;
}

void DeviceKernel(int numBlocks, int sizeBlock, int n, float * p_sum, float * x, float * y, string kernel_name){

  if (kernel_name == "add"){
    KernelAdd <<<numBlocks, sizeBlock>>> (N, d_c, d_a, d_b);  // vector addition on DEVICE
  }
}