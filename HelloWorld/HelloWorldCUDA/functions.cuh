#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

// cuda library
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

// GPU error checking macro
#define gpuErrchk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true){
  if (code != cudaSuccess){
    fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#endif
