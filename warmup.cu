#include "cuda_utils.h"
#include "warmup.h"

void warm_up(int grid, int block)
{
	float * h_data, *d_data;
	h_data = (float *) malloc(grid * block * sizeof(float));
	CUDA_CHECK_ERROR( cudaMalloc( &d_data, grid * block * sizeof(float)) );
	for(int i = 0 ; i < MAX_ITER ; i++)
		warmup<<< grid, block >>>( d_data, grid * block );
	CUDA_CHECK_ERROR( cudaMemcpy( h_data, d_data, grid * block * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
	CUDA_CHECK_ERROR( cudaFree( d_data) );
	free(h_data);
	CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
}

__global__ void warmup(float *data, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	data[i] = i;
	data[i] *= (3.0 * i + N);
}
