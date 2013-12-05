#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <curand_kernel.h>
#include "cuda_utils.h"

#include "tsp.h"
#include "tsp_cuda.h"
#include "warmup.h"

__global__ void tsp_solver(curandState *state, city * cities, unsigned int *s, unsigned int *d, float *ps, float *pd, int *found, float optimal)
{
	*found = -1;
	/* compute current thread id */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	/* store these values for ease. We know this because this is how the kernel parameters are chosen.
	 * If they are chosen differently, we need to pass them as arguments to the kernel */
	int population = gridDim.x;
	int N = blockDim.x; 

	/* tsp algorithm begins here */
	int i = blockIdx.x;
	__shared__ int j;

	/* choose a random 'j' for the every 'i' */
	if(threadIdx.x == 0)
	{
		j = 0;
		while(j == i)
			j = curand(&state[tid]) % population;
	}
	__syncthreads();

	__shared__ unsigned int s_si[MAX_CITIES], s_sj[MAX_CITIES];	// these will hold the two chosen tours i and j
	s_si[threadIdx.x] = s[i*N + threadIdx.x];
	s_sj[threadIdx.x] = s[j*N + threadIdx.x];

	__shared__ unsigned int s_d[MAX_CITIES];			// this will hold the evolved tour from i and j
	/*********************************************************************/
	/* FUNCTION BEGIN : OX() --> inlined for performance */	
	/* Mutate operation:  The mutation algorithm works as follows - 
	 * Find two random indices cut1 and cut2. Rotate tour s1, (cut2 - cut1) times
	 * Replace elements at indices cut1 to cut2 by tour s2.*/

	__shared__ unsigned int cut1, cut2;
	__shared__ int ox_flags[MAX_CITIES], ox_prefix_sum[MAX_CITIES];

	/* Choose cut1 and cut2 such that 0 <= cut1 < cut2 < N */
	if(threadIdx.x == 0)
	{
		unsigned int r1 = 0, r2 = 0;
		while(r1 == r2)
		{
			r1 = curand(&state[tid]) % N;
			r2 = curand(&state[tid]) % N;
		}
		/* Make sure 0 <= cut1 < cut2 < N */
		if(r1 < r2) { cut1 = r1 ; cut2 = r2; }
		else	    { cut2 = r1 ; cut1 = r2; }
	}
	__syncthreads();
	for(int pos = cut1 ; pos < cut2 ; pos++)
	{
		if(s_si[threadIdx.x] == s_sj[pos])
			break;
		ox_flags[threadIdx.x] = (pos > cut2) ? 1 : 0;
	}
	__syncthreads();

		/*********************************************************************/
		/* FUNCTION BEGIN : PrefixSum() --> inlined for performance */	/* TODO : Parallelize !! */
		if(threadIdx.x == 0)
		{
			ox_prefix_sum[cut2] = ox_flags[cut2];
			for(int i = cut2 + 1 ; i < N ; i++)
				ox_prefix_sum[i] = ox_prefix_sum[i-1] + ox_flags[i];
			ox_prefix_sum[0] = ox_prefix_sum[N-1] + ox_flags[0];
			for(int i = 0; i < cut2 ; i++)
				ox_prefix_sum[i] = ox_prefix_sum[i-1] + ox_flags[i];
		}
		__syncthreads();
		/* FUNCTION END : PrefixSum() */
		/*********************************************************************/
	ox_prefix_sum[threadIdx.x]--;	
	if(ox_prefix_sum[threadIdx.x] < (N - cut2))
		ox_prefix_sum[threadIdx.x] += cut2;
	else
		ox_prefix_sum[threadIdx.x] -= (N - cut2);
	__syncthreads();

	if(ox_flags[threadIdx.x])
		s_d[ox_prefix_sum[threadIdx.x]] = s_si[threadIdx.x];
	__syncthreads();

	if((threadIdx.x >= cut1) && (threadIdx.x < cut2))
		s_d[threadIdx.x] = s_sj[threadIdx.x];
	__syncthreads();
	/* FUNCTION END : OX() */
	/*********************************************************************/

	/* FUNCTION BEGIN : 2OPT_best() */
	/* 2OPT local search operation. Find the best path from given tour s_d */
	__shared__ float delta[MAX_CITIES];	/* The reduction in path length caused by the swap */
	__shared__ int j_val[MAX_CITIES]; /* The corresponding value of j for this delta. */
	delta[threadIdx.x] = 0;
	j_val[threadIdx.x] = -1;
	if((threadIdx.x > 0) && (threadIdx.x < N - 2))
	{
		/* We have i = threadIdx.x; We want to calculate
		 * delta = distance(i, i-1) + distance(j+1, j) - distance(i, j+1) - distance(i-1, j); 
		 * Let them denote by d1, d2, d3, d4 respectively. Let the correspondng cities be
		 * icurr, inext, iprev, jcurr, jnext, jprev */
		int i = threadIdx.x;
		city icurr = cities[s_d[i]];
		city iprev = cities[s_d[i-1]];
		// city inext = cities[s_d[(i+1)%N]];
		float d1 = sqrtf((icurr.x - iprev.x) * (icurr.x - iprev.x) + (icurr.y - iprev.y) * (icurr.y - iprev.y));
		float d2, d3, d4;
		city jcurr, jnext;
		jcurr = cities[ s_d[(i+1)%N] ];
		// jprev = cities[ s_d[i] ];
		jnext = cities[ s_d[(i+2)%N] ];
		for(int j = i + 1 ; j < N - 1 ; j++)
		{
			d2 = sqrtf((jnext.x - jcurr.x) * (jnext.x - jcurr.x) + (jnext.y - jcurr.y) * (jnext.y - jcurr.y));
			d3 = sqrtf((icurr.x - jnext.x) * (icurr.x - jnext.x) + (icurr.y - jnext.y) * (icurr.y - jnext.y));
			d4 = sqrtf((iprev.x - jcurr.x) * (iprev.x - jcurr.x) + (iprev.y - jcurr.y) * (iprev.y - jcurr.y));
			float diff = d1 + d2 - d3 - d4;
			if(diff > delta[i])
			{
				delta[i] = diff;
				j_val[i] = j;
			}
			// jprev = jcurr;
			jcurr = jnext;
			jnext = cities[ s_d[(j+1)%N] ];
		}
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		/* Find maximum delta and choose that pair (i,j) for swapping. */
		float max = 0; 
		int maxindex = -1;
		for(int i = 0 ; i < N ; i++)
		{
			if(max > delta[i])
			{
				max = delta[i];
				maxindex = i;
			}
		}
		if((max > 0) && (maxindex >= 0))
		{
			/* swap */
			int choseni = maxindex;
			int chosenj = j_val[maxindex];
			int tempcity = s_d[choseni];
			s_d[choseni] = s_d[chosenj];
			s_d[chosenj] = tempcity;
		}
	}
	__syncthreads();
	/* FUNCTION END : 2OPT_best() */

	/*********************************************************************/

	/* FUNCTION BEGIN : evaluate() */
	/* Now we need to evaluate each path length 
	 * TODO : In current implementation, threadIdx.x == 0 calculates the full path length. 
	 * 	 This could be implemented as add scan */
	__shared__ float p;
	if(threadIdx.x == 0)
	{
		p = 0;
		float dx, dy;
		for(int i = 0 ; i < N ; i++)
		{
			dx = cities[ s_d[(i + 1) % N] ].x - cities[ s_d[i] ].x;
			dy = cities[ s_d[(i + 1) % N] ].y - cities[ s_d[i] ].y;
			p += sqrt((dx*dx) + (dy*dy));
		}
	}
	__syncthreads();
	/* FUNCTION END : evaluate() */
	/*********************************************************************/
	if(p < ps[blockIdx.x])
	{
		d[tid] = s_d[threadIdx.x];
		pd[blockIdx.x] = p;
	}
	else
	{
		d[tid] = s_si[threadIdx.x];
		p = ps[blockIdx.x];
	}

	if(threadIdx.x == 0)
	{
		float err = p - optimal;
		if(err < 0)
			err = -err;
		if(err < (PERCENT_ERROR * optimal))
			*found = blockIdx.x;
	}
}

__global__ void setup_kernel(curandState *state, unsigned long long seed)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	/* Assign same seed to every thread, with a different sequence number and zero offset */
	curand_init(seed, tid, 0, &state[tid]);
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
	city * d_cities;
	int h_found = -1, *d_found;		/* the device updates this to notify the host that acceptable solution was found */

	CUDA_CHECK_ERROR(  cudaMalloc(&d_s1, N * maxpopulation * sizeof(unsigned int)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_s2, N * maxpopulation * sizeof(unsigned int)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_p1, maxpopulation * sizeof(float)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_p2, maxpopulation * sizeof(float)) );
	CUDA_CHECK_ERROR(  cudaMalloc(&d_cities, N * sizeof(city)) );
	
	CUDA_CHECK_ERROR(  cudaMemcpy(d_s1,  s1,   N * maxpopulation * sizeof(unsigned int),  cudaMemcpyHostToDevice) );
	CUDA_CHECK_ERROR(  cudaMemcpy(d_p1,  p1,   maxpopulation * sizeof(float),  cudaMemcpyHostToDevice) );
	CUDA_CHECK_ERROR(  cudaMemcpy(d_cities,  cities,   N * sizeof(city),  cudaMemcpyHostToDevice) );

	CUDA_CHECK_ERROR(  cudaMalloc(&d_found, sizeof(int)) );
	CUDA_CHECK_ERROR(  cudaMemcpy(d_found,  &h_found, sizeof(int),  cudaMemcpyHostToDevice) );

	dim3 grid(maxpopulation);
	dim3 block(N);

	/* setup random number generation in kernel */
	curandState *deviceStates;
	CUDA_CHECK_ERROR(  cudaMalloc((void **) &deviceStates, grid.x * block.x * sizeof(curandState)) );
	setup_kernel<<< grid, block >>> (deviceStates, time(NULL));

	/* warm-up */
	warm_up(grid.x, block.x);


	/* XXX : execute the core genetic algorithm loop to solve TSP */
	int generation = 0;
	while(generation < maxgenerations)
	{
		tsp_solver<<< grid, block >>> (deviceStates, d_cities, d_s1, d_s2, d_p1, d_p2, d_found, optimal);
		CUDA_CHECK_ERROR( cudaMemcpy( &h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost) );
		if(h_found >= 0)
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

	if(h_found >= 0)
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
