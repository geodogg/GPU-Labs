#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
#include <ctime>

// cuda library inlcludes
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>
#include "functions.cuh"

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

// argc - argument count is the number of parameters passed plus one more
//        parameter which is the name of the program that was executed. This is
//        held in the argv[0].
// argv - argument vector
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

    int N = 1 << 17;  // approximately a million elements
    // generate vectors for addition on HOST
    float * c = new float[N]; // allocate memory of million floats on HOST
    if ( c == NULL ) exit (1);  // error check
    float * a = new float[N]; // allocate memory on HOST
    if ( a == NULL ) exit (1);  // error check
    float * b = new float[N]; // allocate memory on HOST
    if ( b == NULL ) exit (1);  // error check

    printline("Check\n")
    // initialize a and b on HOST
    for (int i = 0; i < N; i++){
      a[i] = 1.0f;
      b[i] = 2.0f;
      c[i] = 0.0f;
    }
    printline("Check\n")

    // ~~~~~~~~~~~~~~~ vector addition - DEVICE~~~~~~~~~~~~~~~

    // allocate memory on device for addition
    float * d_c;  // pass the pointer to a float by reference and the address will point to a location on the device after calling cudaMalloc()
    gpuErrchk(cudaMalloc(&d_c, N)); // allocate memory of million floats on DEVICE
    float * d_a;
    gpuErrchk(cudaMalloc(&d_a, N));
    float * d_b;
    gpuErrchk(cudaMalloc(&d_b, N));

    printline("Check\n")

    printf("Number of floats allocated per vector: %d bytes\n", N);

    printf("~~~~~~Vector addition on DEVICE~~~~~~\n");

    // copy input vectors at c, a and b to DEVICE memory locations at d_a and d_b
    gpuErrchk(cudaMemcpy( d_a, a, N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy( d_b, b, N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy( d_c, c, N, cudaMemcpyHostToDevice));

    printline("Check\n")

    clock_t tic = clock();  // start clocking

    printline("Check\n")

    int sizeBlock = 256;
    int numBlocks = 1;
//  scanf("Please enter number of threads per block:%i\n", sizeBlock);
//  numBlocks = (N + sizeBlock - 1) / sizeBlock;

    printline("Check\n")

    DeviceKernel( numBlocks, sizeBlock, N, d_c, d_a, d_b);  // vector addition on DEVICE

    printline("Check\n")

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printline("Check\n")

    // Copy result from device memory to host memory
    gpuErrchk(cudaMemcpy( c, d_c, N, cudaMemcpyDeviceToHost));

    printf("Addition of a[0] b[0] equals %f.\n", c[0]); // print single element output

    clock_t toc = clock() - tic;
    float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;

    printline("Check\n")

    printf("Vector addition on the HOST\nElapsed time: %f (sec)\n", elapsed_time);

    // free memory on HOST
    delete [] c;
    delete [] a;
    delete [] b;

    // free memory on DEVICE
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    fclose(outfile);
    return 0;
}
