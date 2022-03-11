#include "cuda_sobelf.h"

#include <cuda_runtime.h>
#include <cstdio>

#define numBlocks 64
#define threadsPerBlock 64

__device__ int is_blur_done;

#define CONV(l, c, nb_c) \
    (l) * (nb_c) + (c)

__global__ void apply_blur_filter_kernel(pgrey *p, int delta_p, pgrey *temp, int width, int height, int size, int threshold)
{
    p += delta_p;

    int first_i = blockIdx.x * blockDim.x + threadIdx.x;
    int taille = width * height;
    int delta = gridDim.x * blockDim.x;

    // copy from p to temp
    int i = first_i;
    while (i < taille)
    {
        temp[i] = p[i];
        i += delta;
    }
    __threadfence();
    
    i = first_i;
    while (i < taille)
    {
        int k = i % width;
        int j = i / width;

        if(j >=  height - size)
            break;
        if(k < size || k >= width - size || j < size){
            i += delta;
            continue;
        }

        int stencil_j, stencil_k;
        int t = 0;

        for (stencil_j = -size; stencil_j <= size; stencil_j++)
        {
            for (stencil_k = -size; stencil_k <= size; stencil_k++)
            {
                t += p[CONV(j + stencil_j, k + stencil_k, width)];
            }
        }

        temp[i] = t / ((2 * size + 1) * (2 * size + 1));

        i += delta;
    }

    __threadfence();
    

    i = first_i;
    while (i < taille)
    {
        int diff = p[i] - temp[i];
        if(diff > threshold || diff < -threshold){
            is_blur_done = 0;
        }
        p[i] = temp[i];
        i += delta;
    }
}

void apply_filter_cuda(pgrey *p, int width, int height, int position, int size, int threshold)
{
    if(position == 0)return;
    pgrey *d_p, *d_temp;
    cudaMalloc((void **)&d_p, width * height * sizeof(pgrey));
    cudaMemcpy(d_p, p, width * height * sizeof(pgrey), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_temp, width * height * sizeof(pgrey));

    if(position != 0){
        // we need to blur this part
        int done;
        int n_iter = 0;
        do
        {
            done = 1;
            cudaMemcpyToSymbol(is_blur_done, &done, sizeof(int));
            int delta_p = 0;
            int blur_height = height-1;
            if(position == 2){
                delta_p = width;
            }
            apply_blur_filter_kernel<<<numBlocks,threadsPerBlock>>>(d_p, delta_p, d_temp, width, blur_height, size, threshold);
            cudaMemcpyFromSymbol(&done, is_blur_done, sizeof(int));
            n_iter++;
        } while (!done);
    }

    cudaMemcpy(p, d_p, width * height * sizeof(pgrey), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_p);
}