#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <ctime>

// vector addition on HOST
void add(int n, float * p_sum, float * x, float * y);

// GPU error checking macro
#define gpuErrchk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true){
  if (code != cudaSuccess){
    fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#endif
