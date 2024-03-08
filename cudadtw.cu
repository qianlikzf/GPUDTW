/*  
 * Copyright (C) 2024 Wuhan University of Technology
 * Authors: Wang Zihao <qianlkzf@outlook.com> 
 *  
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, either version 3 of the License, or  
 * (at your option) any later version.  
 *  
 * This program is distributed in the hope that it will be useful,  
 * but WITHOUT ANY WARRANTY; without even the implied warranty of  
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
 * GNU General Public License for more details.  
 *  
 * You should have received a copy of the GNU General Public License  
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.  
 */
#include <stdio.h>
#include <stdlib.h>

extern __shared__ float array[];

__global__ void calc_dtw (
    unsigned SRC_LEN,
    unsigned TRG_LEN,
    unsigned TRG_COT,
    float *S, 
    float *TT, 
    float *Result)
{
    // blockDim.x = TRG_LEN*TRG_COT
    float* path_h1 = (float*)array;
    float* path_h2 = (float*)&path_h1[blockDim.x];
    float* dist    = (float*)&path_h2[blockDim.x];

    //int blockId = (blockIdx.y*gridDim.x + blockIdx.x);
    //int G_idx   = blockId *blockDim.x +threadIdx.x;
    float *T =     TT + (blockIdx.y*gridDim.x +blockIdx.x) *blockDim.x;
    float *R = Result + (blockIdx.y*gridDim.x +blockIdx.x) *TRG_COT;
    
    float *ex;
    int i,j;
    
    int sub_x = threadIdx.x % TRG_LEN;
    int x_cot = threadIdx.x / TRG_LEN;
    
    // first line speical, do first
    // 1. paralle, first line's every element's dist
    dist[threadIdx.x] = (S[0] -T[threadIdx.x])
                       *(S[0] -T[threadIdx.x]);
    __syncthreads();
    // 2. serie, first line's every element's serie's dist
    if (sub_x == 0){
        path_h1[threadIdx.x] = dist[threadIdx.x];
        for (i=1; i <TRG_LEN; i++) {
            path_h1[i +threadIdx.x] = path_h1[i-1 +threadIdx.x]
                                    +    dist[i   +threadIdx.x];
        }
    }
    __syncthreads();
    
    for (i=1; i <SRC_LEN; i++){ // do circle
        // 1. paralle, calc itself's DISTANCE, for speed follow progress
        //    use memeory to rise speed
        dist[threadIdx.x] = (S[i] -T[threadIdx.x])
                           *(S[i] -T[threadIdx.x]);
        __syncthreads();
        // 2. paralle, get from upper line's "up","left-up"- the min dist
        if (sub_x == 0) // FIRST element speical, add DISTANCE here
            path_h2[threadIdx.x] =  path_h1[threadIdx.x] + dist[threadIdx.x];
        else
            path_h2[threadIdx.x] = min (path_h1[threadIdx.x],
                                        path_h1[threadIdx.x-1]);
        __syncthreads();
        // 2. serie, compare to left(front) element with myself, get the less value
        if (sub_x == 0) // the first element had plused dist, so can use here
            for (j=1; j<TRG_LEN; j++) 
                path_h2[j +threadIdx.x] = min(path_h2[j   +threadIdx.x], 
                                              path_h2[j-1 +threadIdx.x])
                                        + dist[j +threadIdx.x];
        __syncthreads();
        ex      = path_h2;
        path_h2 = path_h1;
        path_h1 = ex;
        __syncthreads();
    }
    // when all done, can return the result
    if (sub_x == 0){
        R[x_cot] = sqrt(ex[threadIdx.x +TRG_LEN -1]);
    }
}    
