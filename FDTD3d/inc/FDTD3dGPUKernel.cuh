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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
#define RADIUS 4

__constant__ float stencil[RADIUS + 1];

__global__ void FiniteDifferencesKernel(float *output,
                                        float *outputFULL,
                                        const float *input,
                                        const int dimx,
                                        const int dimy,
                                        const int dimz,
                                        DEVICES *arr_device)
{
    bool validr = true;
    bool validw = true;
    const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;   // device global grid index x
    const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;   // device global grid index y
    const int ltidx = threadIdx.x;  // thread index x in the block
    const int ltidy = threadIdx.y;  // thread index y in the block
    const int workx = blockDim.x;   // block index x in the grid
    const int worky = blockDim.y;   // block index y in the grid
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

    int inputIndex  = 0;
    int outputIndex = 0;
    int otherGPUinputIndex = 0;

    float infront[RADIUS];
    float behind[RADIUS];
    float current;

    const int tx = ltidx + RADIUS;
    const int ty = ltidy + RADIUS;

    bool prev_device = false;
    bool GPUfirst = false;
    bool GPUmiddle = false;
    bool GPUlast = false;
    int num_d = arr_device[0].num_devices;
    int current_device = 0;
    cudaGetDevice(&current_device);

    if (num_d > 1)
    {
        if (current_device == 0)
        {
            GPUfirst = true;
            arr_device[current_device].stride_y = dimx + 2 * RADIUS;                                              // the stride in the y direction
            arr_device[current_device].stride_z = arr_device[current_device].stride_y * (dimy + RADIUS);          // the stride in the z direction
            inputIndex += RADIUS * arr_device[current_device].stride_y + RADIUS;                                  // Advance inputIndex to start of inner volume
            inputIndex += gtidy * arr_device[current_device].stride_y + gtidx;                                    // Advance inputIndex to target element
        }
        else if(current_device + 1 == num_d)
        {
            GPUlast = true;
            arr_device[current_device].stride_y = dimx + 2 * RADIUS;                                                // the stride in the y direction
            arr_device[current_device].stride_z = arr_device[current_device].stride_y * (dimy + RADIUS);            // the stride in the z direction
            inputIndex += RADIUS;                                                                                   // Advance inputIndex to start of data
            inputIndex += gtidy * arr_device[current_device].stride_y + gtidx;                                      // Advance inputIndex to target element
        }
        else
            GPUmiddle = true;
            arr_device[current_device].stride_y = dimx + 2 * RADIUS;                               // the stride in the y direction
            arr_device[current_device].stride_z = arr_device[current_device].stride_y * dimy;      // the stride in the z direction
            inputIndex += RADIUS;                                                                  // Advance inputIndex to start of inner volume
            inputIndex += gtidy * arr_device[current_device].stride_y + gtidx;                     // Advance inputIndex to target element
    }
    else
    {
        arr_device[current_device].stride_y = dimx + 2 * RADIUS;                               // the stride in the y direction
        arr_device[current_device].stride_z = arr_device[current_device].stride_y * (dimy + 2 * RADIUS);                  // the stride in the z direction
        inputIndex += RADIUS * arr_device[current_device].stride_y + RADIUS;                   // Advance inputIndex to start of inner volume
        inputIndex += gtidy * arr_device[current_device].stride_y + gtidx;                     // Advance inputIndex to target element
    }

    // Check in bounds
    if (gtidx >= dimx + RADIUS)
        validr = false;
    else if ((gtidy >= dimy + RADIUS) && num_d == 1)
        validr = false;
    else if ((gtidy >= dimy + RADIUS) && GPUlast)
        validr = false;

    if (gtidx >= dimx)
        validw = false;
    else if ((gtidy >= dimy) && num_d == 1)
        validw = false;
    else if ((gtidy >= dimy) && GPUlast)
        validw = false;

    // do original method for first gpu or one gpu case
    // Preload the "infront" and "behind" data
    for (int i = RADIUS - 2 ; i >= 0 ; i--)
    {
        if (validr)
            behind[i] = input[inputIndex];

        inputIndex += arr_device[current_device].stride_z;
    }

    if (validr)
        current = input[inputIndex];

    outputIndex = inputIndex;
    inputIndex += arr_device[current_device].stride_z;

    for (int i = 0 ; i < RADIUS ; i++)
    {
        if (validr)
            infront[i] = input[inputIndex];

        inputIndex += arr_device[current_device].stride_z;
    }


    // Step through the xy-planes
#pragma unroll 9

    for (int iz = 0 ; iz < dimz ; iz++)
    {
        // Advance the slice (move the thread-front)
        for (int i = RADIUS - 1 ; i > 0 ; i--)
            behind[i] = behind[i - 1];

        behind[0] = current;
        current = infront[0];
#pragma unroll 4

        for (int i = 0 ; i < RADIUS - 1 ; i++)
            infront[i] = infront[i + 1];

        if (validr)
            infront[RADIUS - 1] = input[inputIndex];

        inputIndex  += arr_device[current_device].stride_z;
        outputIndex += arr_device[current_device].stride_z;
        // if (prev_device)
        //     otherGPUinputIndex += arr_device[current_device - 1].stride_z;

        cg::sync(cta);

        // Note that for the work items on the boundary of the problem, the
        // supplied index when reading the halo (below) may wrap to the
        // previous/next row or even the previous/next xy-plane. This is
        // acceptable since a) we disable the output write for these work
        // items and b) there is at least one xy-plane before/after the
        // current plane, so the access will be within bounds.

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < RADIUS)
        {
            //
            // // obtain initial 4 data points from previous device
            // if ((GPUlast || GPUmiddle) && (gtidy <= RADIUS - 1))
            // {
            //     // Preload the "infront" and "behind" data
            //     prev_device = true;
            //     for (int i = RADIUS - 2 ; i >= 0 ; i--)
            //     {
            //         otherGPUinputIndex = (arr_device[current_device - 1].stride_z - arr_device[current_device - 1].stride_y) - (RADIUS - 1 - gtidy) * arr_device[current_device - 1].stride_y + gtidx + RADIUS;
            //         if (validr)
            //             behind[i] = arr_device[current_device - 1].d_in[otherGPUinputIndex];
            //
            //         otherGPUinputIndex += arr_device[current_device - 1].stride_z;
            //     }
            //     if (validr)
            //         current = input[inputIndex];
            //
            //     outputIndex = inputIndex;
            //     inputIndex += arr_device[current_device].stride_z;
            //
            //     for (int i = 0 ; i < RADIUS ; i++)
            //     {
            //         if (validr)
            //             infront[i] = input[inputIndex];
            //
            //         inputIndex += arr_device[current_device].stride_z;
            //     }
            // }

            // if ((GPUlast || GPUmiddle) && (gtidy <= RADIUS - 1))
            // {
            //
            // }

            tile[ltidy][tx]                  = input[outputIndex - RADIUS * arr_device[current_device].stride_y];
            tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * arr_device[current_device].stride_y];

            // //IF GPU0 and gtidy<dimy-1
            // tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
            // //else WHAT ADDRESS FROM THE OTHER GPU ???????????????
            // tile[ltidy + worky + RADIUS][tx] = otherGPUinputIndex[???????????];
        }

        // Halo left & right
        if (ltidx < RADIUS)
        {
            tile[ty][ltidx]                  = input[outputIndex - RADIUS];
            tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
        }

        tile[ty][tx] = current;
        cg::sync(cta);

        // Compute the output value
        float value = stencil[0] * current;
#pragma unroll 4

        for (int i = 1 ; i <= RADIUS ; i++)
        {
            value += stencil[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (validw){
            output[outputIndex] = value;
            outputFULL[outputIndex + current_device * arr_device[0].volumeSizeOffset] = value;
        }
    }
}
