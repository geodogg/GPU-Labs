#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
#include <ctime>

// cuda library inlcludes
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <cooperative_groups.h>

using namespace std;

// GPU error checking macro
#define gpuErrchk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true){
  if (code != cudaSuccess){
    fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// outputs debugging macro
#define printline(ans) { fprintf(outfile, "file: %s line: %d\n - ", __FILE__, __LINE__); fprintf(outfile, ans); }


#define N (1 << 20)
#define sizeBlock (1024)
#define numBlocks (1024)

// initialize unified memory
__device__ __managed__ float c[ (int) 1 << 20];

// addition kernel
__global__
void KernelAdd(){

  __shared__ float a = 0.10;
  __shared__ float b = 0.01;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride)
      c[i] = a + b + i;

  // int i = blockIdx.x * blockDim.x + threadIdx.x;
  // if ( i < N )
  //   c[i] = a + b + i;

}

// subtraction kernel
__global__
void KernelSub(){

  __shared__ float a = 0.10;
  __shared__ float b = 0.01;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride)
      c[i] = c[i] - a - b - i;

  // int i = blockIdx.x * blockDim.x + threadIdx.x;
  // if ( i < N )
  //   c[i] = a + b + i;

}

int main(int argc, char * argv[]){
    printf("Running program: %s\n", argv[0]);

    // outfile for debugging
    FILE * outfile;
    outfile = fopen("debug.txt", "w");
    if (outfile == NULL){
      printf(".....there is an error opening debug file....\n");
      return 0;
    }

    printline("Hello! Welcome to the HelloWorld equivalent of CUDA.\n")

//  scanf("Please enter number of threads per block:%i\n", sizeBlock);

    clock_t tic = clock();  // start clocking

    printline("Check\n")

    int p = 0;
    for(int i = 0; i < N; p++, i+=(1<<p))
      fprintf(outfile, "c[%d] = %f\n", i, c[i]);

    cudaSetDevice(1);

    KernelAdd<<< numBlocks, sizeBlock >>> ();
    cudaDeviceSynchronize();

    clock_t toc = clock() - tic;
    float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;   // finish clocking

    printline("Check\n")

    printf("Vector addition on the DEVICE 0\nElapsed time: %f (sec)\n", elapsed_time);

    printline("Check\n")
    p = 0;
    for(int i = 0; i < N; p++, i+=(1<<p))
      fprintf(outfile, "c[%d] = %f\n", i, c[i]);

    cudaSetDevice(0);

    tic = clock();  // start clocking

    cudaDeviceSynchronize();
    KernelSub<<< numBlocks, sizeBlock >>> (N, a, b);

    toc = clock() - tic;
    elapsed_time = ((float)toc) / CLOCKS_PER_SEC;   // finish clocking

    printf("Vector addition on the DEVICE 0\nElapsed time: %f (sec)\n", elapsed_time);

    printf("Vector subtration on the DEVICE 1\n");

    printline("Check\n")
    p = 0;
    for(int i = 0; i < N; p++, i+=(1<<p))
      fprintf(outfile, "c[%d] = %f\n", i, c[i]);

    fclose(outfile);
    return 0;
}
