  /*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _FDTD3DGPU_H_
#define _FDTD3DGPU_H_

#include <cstddef>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

#define k_blockDimX    32
#define k_blockDimMaxY 16
#define k_blockSizeMin 128
#define k_blockSizeMax (k_blockDimX * k_blockDimMaxY)

#include <cuda_runtime.h>

typedef enum { first, middle, last } gpu_location;

typedef struct
{
    int device;                     // device ID
    size_t data_size_device;        // bytes of data to be processed on device
    size_t data_size_total;         // bytes of data to be processed by program
    size_t padded_data_size_device; // padding of current device data that FDTD3d utilizes
    size_t padded_data_size_total;  // padding of the total data
    float *d_out;          // pointer to device output data
    float *d_in;           // pointer to device input data
    float *h_out;          // pointer to host location of data
    float *in_data;        // pointer to input d_data
    dim3 dimBlock;         // block dimensions of GPU
    dim3 dimGrid;          // grid dimensions of GPU
    cudaDeviceProp deviceProp;  // cuda device properties
    int num_devices;            // number of devices
    int stride_z;               // the stride used to get to the next layer of the input data at a given xy point
    int stride_y;               // the stride used to get to the next y location at a given x point
    int startingIndex;          // starting index of current devices
    int endingIndex;            // i.e. along the gtidx = Radius index on first z layer
    gpu_location gpu_case;      // gpu_case indicates which gpu location the current gpu is at

} DEVICES;

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv);
bool fdtdGPU(cudaStream_t *streams, DEVICES *arr_device, float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv);
//bool compareDataSmall(const float *output, const float *input, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance);

#endif
