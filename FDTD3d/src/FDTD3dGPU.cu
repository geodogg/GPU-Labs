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

#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "FDTD3dGPUKernel.cuh"

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    // Save the result
    *result = (memsize_t)memsize;
    return true;
}

bool fdtdGPU(cudaStream_t *streams, DEVICES *arr_device, float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz / arr_device[0].num_devices;
    const size_t      volumeSize_device = dimx * dimy * dimz / arr_device[0].num_devices;

    // int               deviceCount  = 0;
    // int               targetDevice = 0;
    // float            *bufferOut    = 0;
    // float            *bufferIn     = 0;
    // dim3              dimBlock;
    // dim3              dimGrid;

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;

    for (int i = 0; i < arr_device[0].num_devices; i++){
        arr_device[i].paddedVolumeSize = arr_device[i].data_size_device + padding;
    }


#ifdef GPU_PROFILING
    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    const int profileTimesteps = timesteps - 1;

    if (profileTimesteps < 1)
    {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
    }

#endif

    // Check the radius is valid
    if (radius != RADIUS)
    {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    // // Get the number of CUDA enabled GPU devices
    // checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    //
    // // Select target device (device 0 by default)
    // targetDevice = findCudaDevice(argc, (const char **)argv);
    //
    // checkCudaErrors(cudaSetDevice(targetDevice));
    //
    // // Allocate memory buffers

     // allocate device data. split equally among GPUs
    for (int i = 0; i < arr_device[0].num_devices; i++){
        float *ptr_out = 0;
        float *ptr_in = 0;

        // set cuda device
        checkCudaErrors(cudaSetDevice(arr_device[i].device));
        // set input device data
        checkCudaErrors(cudaMalloc((void **) (&(ptr_out)), arr_device[i].paddedVolumeSize * sizeof(float)));
        arr_device[i].d_out = ptr_out;
        // set output device data
        checkCudaErrors(cudaMalloc((void **) (&(ptr_in)), arr_device[i].paddedVolumeSize * sizeof(float)));
        arr_device[i].d_in = ptr_in;
    }

    // // allocate device data. split equally among GPUs
    // for (int i = 0; i < arr_device[0].num_devices; i++){
    //     // set cuda device
    //     checkCudaErrors(cudaSetDevice(arr_device[i].device));
    //     // set input device data
    //     checkCudaErrors(cudaMalloc((void **)&(arr_device[i].d_out), (paddedVolumeSize * sizeof(float)) / (arr_device[0].num_devices)));
    //     // set output device data
    //     checkCudaErrors(cudaMalloc((void **)&(arr_device[i].d_in), (paddedVolumeSize * sizeof(float)) / (arr_device[0].num_devices)));
    // }

    // Check for a command-line specified block size
    int userBlockSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "block-size"))
    {
        userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
        // Constrain to a multiple of k_blockDimX
        userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

        // Constrain within allowed bounds
        userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
    }
    else
    {
        userBlockSize = k_blockSizeMax;
    }

    // checkCudaErrors(cudaSetDevice(arr_device[i].device));
    // // Check the device limit on the number of threads
    // struct cudaFuncAttributes funcAttrib;
    // checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));
    //
    // userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);
    //
    // // Set the block sizes
    // dimBlock.x = k_blockDimX;
    // // Visual Studio 2005 does not like std::min
    // //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);
    // dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
    // dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    // dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    // printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    // printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);
    //
    // // Check the block size is valid
    // if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    // {
    //     printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
    //     exit(EXIT_FAILURE);
    // }

    // set gpu architecture abstrations
    for (int i = 0; i < arr_device[0].num_devices; i++){

        checkCudaErrors(cudaSetDevice(arr_device[i].device));

        // Check the device limit on the number of threads
        struct cudaFuncAttributes funcAttrib;
        checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

        userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

        // Set the block sizes
        arr_device[i].dimBlock.x = k_blockDimX;
        arr_device[i].dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
        arr_device[i].dimGrid.x = (unsigned int)ceil((float)dimx / arr_device[i].dimBlock.x);
        arr_device[i].dimGrid.y = (unsigned int)ceil((float)dimy / arr_device[i].dimBlock.y);
        printf(" for device %d\n", arr_device[i].device);
        printf(" set block size to %dx%d\n", arr_device[i].dimBlock.x, arr_device[i].dimBlock.y);
        printf(" set grid size to %dx%d\n", arr_device[i].dimGrid.x, arr_device[i].dimGrid.y);

        // Check the block size is valid
        if (arr_device[i].dimBlock.x < RADIUS || arr_device[i].dimBlock.y < RADIUS)
        {
            printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", arr_device[i].dimBlock.x, arr_device[i].dimBlock.y, RADIUS);
            exit(EXIT_FAILURE);
        }
    }

    int offset = 0;
    for (int i = 0; i < arr_device[0].num_devices; i++){

        checkCudaErrors(cudaSetDevice(arr_device[i].device));

        // Copy the input to the device input buffer
        checkCudaErrors(cudaMemcpy(arr_device[i].d_in + padding, input + offset, arr_device[i].data_size_device * sizeof(float), cudaMemcpyHostToDevice));

        // Copy the input to the device output buffer (actually only need the halo)
        checkCudaErrors(cudaMemcpy(arr_device[i].d_out + padding, input + offset, arr_device[i].data_size_device * sizeof(float), cudaMemcpyHostToDevice));

        offset += (volumeSize);

    }

    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

    // Copy the coefficients to the device coefficient buffer

    // // Copy the input to the device input buffer
    // checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));
    //
    // // Copy the input to the device output buffer (actually only need the halo)
    // checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));
    //
    // // Copy the coefficients to the device coefficient buffer
    // checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

#ifdef GPU_PROFILING
    // Create the events
    checkCudaErrors(cudaEventCreate(&profileStart));
    checkCudaErrors(cudaEventCreate(&profileEnd));
#endif

#ifdef GPU_PROFILING
    // Enqueue start event
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

    // pad addresses.
    for (int i = 0; i < arr_device[0].num_devices; i++){
        arr_device[i].d_in += padding;
        arr_device[i].d_out += padding;
    }

    printf(" GPU FDTD loop\n");

    for (int it = 0 ; it < timesteps ; it++)
    {
        printf("\tt = %d ", it);

        // Launch the kernel
        printf("launch kernel\n");

        checkCudaErrors(cudaSetDevice(100));


        for (int i = 0; i < arr_device[0].num_devices; i++){

            printf("for device %d\n", arr_device[i].device);

            checkCudaErrors(cudaSetDevice(arr_device[i].device));

            FiniteDifferencesKernel<<<arr_device[i].dimGrid, arr_device[i].dimBlock, 0, streams[i]>>>(arr_device[i].d_out, arr_device[i].d_in, dimx, dimy, dimz / arr_device[0].num_devices);

            float *tmp = arr_device[i].d_out;
            arr_device[i].d_out = arr_device[i].d_in;
            arr_device[i].d_in = tmp;

        }

        // Toggle the buffers
        // Visual Studio 2005 does not like std::swap
        //    std::swap<float *>(bufferSrc, bufferDst);
        // float *tmp = bufferDst;
        // bufferDst = bufferSrc;
        // bufferSrc = tmp;
    }

    // run this. There was error with scaling up to 4 gpus. already published on git
    checkCudaErrors(cudaSetDevice(100));


    printf("\n");


#ifdef GPU_PROFILING
    // Enqueue end event
    checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif
    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkCudaErrors(cudaMemcpy(output, arr_device[0].d_out, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Report time
#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0)
    {
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0)
    {
        // Convert milliseconds to seconds
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, arr_device[0].dimBlock.x * arr_device[0].dimBlock.y);
    }
#endif

    // Cleanup
    if (arr_device[0].d_in)
    {
        checkCudaErrors(cudaFree(arr_device[0].d_in));
    }

    if (arr_device[0].d_out)
    {
        checkCudaErrors(cudaFree(arr_device[0].d_out));
    }

#ifdef GPU_PROFILING

    if (profileStart)
    {
        checkCudaErrors(cudaEventDestroy(profileStart));
    }

    if (profileEnd)
    {
        checkCudaErrors(cudaEventDestroy(profileEnd));
    }

#endif
    return true;
}
