#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
#include <ctime>

// cuda library inlcludes
#include <cuda_runtime.h>
#include <helper_cuda.h>
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

// initialize unified memory
__device__ __managed__ float c[ (int) 1 << 20];

// addition kernel
__global__
void KernelAdd(int N, float a, float b){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < N )
    c[i] = a + b + i;

  // using single block
  // c[threadIdx.x] = a + b + threadIdx.x;
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

    int N = 1 << 20;  // 1024 elements

    float a = 0.10;
    float b = 0.01;

    int sizeBlock = 1024;
    int numBlocks = 1024;
//  scanf("Please enter number of threads per block:%i\n", sizeBlock);

    clock_t tic = clock();  // start clocking

    clock_t toc = clock() - tic;
    float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;

    printline("Check\n")

    KernelAdd<<< numBlocks, sizeBlock >>> (N, a, b);
    cudaDeviceSynchronize();

    printline("Check\n")

    printf("Vector addition on the DEVICE\nElapsed time: %f (sec)\n", elapsed_time);

    // Wait for GPU to finish before accessing on host
//    cudaDeviceSynchronize();
    // waiting for Device to synchronize on cedar or busy remote servers will cause
    // program to take forever.

    printline("Check\n")

    for(int i = 0; i < N; i++)
      fprintf(outfile, "c[%d] = %f\n", i, c[i]);

    fclose(outfile);
    return 0;
}
