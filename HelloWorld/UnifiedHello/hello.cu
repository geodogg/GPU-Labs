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
int N = 1 << 10;  // 1024 elements
__device__ __managed__ float c[N];

// addition kernel
__global__
void KernelAdd(int N, float a, float b){

  // int i = blockIdx.x * blockDim.x + threadIdx.x;
  // if ( i < N )
  //   c[i] = a + b + i;

  // using single block
  c[threadIdx.x] = a + b + threadIdx.x;
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

    float a = 1.0;
    float b = 2.0;

    int sizeBlock = 256;
    int numBlocks = 1;
//  scanf("Please enter number of threads per block:%i\n", sizeBlock);

    clock_t tic = clock();  // start clocking

    printline("Check\n")

    KernelAdd<<< numBlocks, sizeBlock >>> (N, a, b);

    printline("Check\n")

    // Wait for GPU to finish before accessing on host
//    cudaDeviceSynchronize();
    // waiting for Device to synchronize on cedar or busy remote servers will cause
    // program to take forever.
    int i = 0;
    for(auto iter = c.begin(); iter != c.end(); iter++, i++)
      fprintf(outfile, "c[%d] = %f", i, (*iter);

    printline("Check\n")

    clock_t toc = clock() - tic;
    float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;

    printf("Vector addition on the DEVICE\nElapsed time: %f (sec)\n", elapsed_time);

    fclose(outfile);
    return 0;
}
