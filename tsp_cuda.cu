#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_utils.h"

#include "tsp.h"
#include "tsp_cuda.h"
#include "warmup.h"

__global__ void tsp_solver(unsigned int *s, unsigned int *d, float *ps, float *pd, int *found)
{
	/* temporarily writing a dummy kernel that just copies s into d */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	d[tid] = s[tid];
}

int run(city * cities, int N, int maxgenerations, int maxpopulation, float optimal, unsigned int * result_tour)
{
	unsigned int *s1, *tour;
	float *p1;
	/* s1 is actually a 2-D matrix modeled as 1-D array. Its rows represent each individual (a tour).
	 * and the elements in row represents each city. Thus, each row contain N unique elements from 0..N-1
	 * which represent the city index to be visited. This index corresponds with the original "cities"
	 * list. e.g. if s1[i*N + 0] = 3 and s1[i*N + 1] = 5, then it means that the ith tour starts with city
	 * represented by cities[3] and the next city visited is cities[5] and so on..
	 * The number of such rows (the values of i) = maxpopulation */
	s1 = (unsigned int *)  malloc(N * maxpopulation * sizeof(unsigned int));

	/* p1 is the tour length of each tour in s1. Thus, the number of elements in p1 = maxpopulation */
	p1 = (float *) malloc(maxpopulation * sizeof(float));

	int i,j;
	/* initialize population with random tours */
	for(i = 0 ; i < maxpopulation ; i++)
	{
		tour = (s1 + i*N);
		generate_random_tour(tour, N);
		p1[i] = tour_length(tour, N, cities);
		if(optimal >= 0)	/* an optimal solution has been provided. */
		{
			if(is_acceptable(tour, N, cities, optimal))
			{
				for(j = 0 ; j < N ; j++)
					result_tour[j] = tour[j];
				return 0;
			}
		}
	}

	/* Now initialize the device memory. These are corresponding device pointers for s1 and p1.
	 * We have two copies of them so that we can deal with 2 generations at a time (parents and their offsprings).
	 * These would be used similar to double buffering so that they alternate between every generation.
	 * e.g. d_s1 have parents while d_s2 have their offsprings/mutations during 1st gen; 
	 *      d_s2 have parents while d_s1 have their offsprings/mutations during 2nd gen; and so on..
	 */
	unsigned int *d_s1, *d_s2, *d_stemp;
	float *d_p1, *d_p2, *d_ptemp;
	int h_found = -1, *d_found;		/* the device updates this to notify the host that acceptable solution was found */

	CUDA_CHECK_ERROR(  cudaMalloc(&d_s1, N * maxpopulation * sizeof(unsigned int)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_s2, N * maxpopulation * sizeof(unsigned int)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_p1, maxpopulation * sizeof(float)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_p2, maxpopulation * sizeof(float)) );
	
	CUDA_CHECK_ERROR(  cudaMemcpy(d_s1,  s1,   N * maxpopulation * sizeof(unsigned int),  cudaMemcpyHostToDevice) );
	CUDA_CHECK_ERROR(  cudaMemcpy(d_p1,  p1,   maxpopulation * sizeof(float),  cudaMemcpyHostToDevice) );

	CUDA_CHECK_ERROR(  cudaMalloc(&d_found, sizeof(int)) );
	CUDA_CHECK_ERROR(  cudaMemcpy(d_found,  &h_found, sizeof(int),  cudaMemcpyHostToDevice) );

	dim3 grid(maxpopulation);
	dim3 block(N);

	/* warm-up */
	warm_up(maxpopulation, N);

	/* XXX : execute the core genetic algorithm loop to solve TSP */
	int generation = 0;
	while(generation < maxgenerations)
	{
		tsp_solver<<< grid, block >>> (d_s1, d_s2, d_p1, d_p2, d_found);
		CUDA_CHECK_ERROR( cudaMemcpy( &h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost) );
		if(h_found > 0)
			break;

		/* swap d_s1 and d_s2 */
		d_stemp = d_s1;
		d_s1 = d_s2;
		d_s2 = d_stemp;

		/* swap d_p1 and d_p2 */
		d_ptemp = d_p1;
		d_p1 = d_p2;
		d_p2 = d_ptemp;

		/* increment the generation */
		generation++;
	}

	if(h_found > 0)
		fprintf(stderr, "Optimal solution was found\n");
	else
		fprintf(stderr, "Optimal solution was NOT found\n");

	/* Copy back the results in any case */
	CUDA_CHECK_ERROR( cudaMemcpy( s1, d_s2, N * maxpopulation * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	CUDA_CHECK_ERROR( cudaMemcpy( p1, d_p2, maxpopulation * sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	/* clean up device memory */
	CUDA_CHECK_ERROR( cudaFree(d_s1) );
	CUDA_CHECK_ERROR( cudaFree(d_s2) );
	CUDA_CHECK_ERROR( cudaFree(d_p1) );
	CUDA_CHECK_ERROR( cudaFree(d_p2) );
	CUDA_CHECK_ERROR( cudaFree(d_found) );

	/* send the result back to driver */
	if(h_found > 0)
	{
		for(j = 0 ; j < N ; j++)
			result_tour[j] = (s1 + h_found * N)[j];
	}
	else
	{
		/* Optimal solution was not found.
		 * TODO : handle this effectively and send so far best tour found,
		 *	  currently returning the zeroth tour */
		for(j = 0 ; j < N ; j++)
			result_tour[j] = s1[j];
	}

	return generation;
}

void generate_random_tour(unsigned int *tour, int size)
{
	/* Each tour should have 'size' number of unique elements in some random sequence.
	 * We use different seeds for different tours so than the pseudo-random number
	 * generator will not have same results for all tours */

	/* set seed depending on current time (number of seconds since epoch) */
	srandom(time(NULL));
	
	/* generate (pseudo-)random tour */
	int i;
	unsigned int r;
	for(i = 0 ; i < size ; i++)
	{
		r = random() % size;
		while(is_present(tour, r, i))			/* TODO : pretty crude way of generating unique numbers in 0..N. */
			r = random() % size;
		tour[i] = r;
	}
}

int is_present(unsigned int * haystack, unsigned int needle, int size)		// This is simple linear search. TODO : binary search if reqd.
{
	int i;
	for(i = 0 ; i < size ; i++)
	{
		if(haystack[i] == needle)
			return 1;
	}
	return 0;
}

int is_acceptable(unsigned int * tour, int N, city * cities, float optimal)
{
	float len = tour_length(tour, N, cities);
	float delta = len - optimal;
	delta = (delta < 0) ? -delta : delta;
	float allowed_err = optimal * PERCENT_ERROR / 100.0;
	if(delta < allowed_err)
		return 1;
	else
		return 0;
}
