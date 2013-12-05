#ifndef __TSP_CUDA_H
#define __TSP_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tsp.h"

__global__ void tsp_solver(curandState *state, city * cities, unsigned int *s, unsigned int *d, float *ps, float *pd, int *found, float optimal);
__global__ void setup_kernel(curandState *state, unsigned long long seed);

void generate_random_tour(unsigned int *tour, int size);
int is_present(unsigned int * haystack, unsigned int needle, int size);
int is_acceptable(unsigned int * tour, int N, city * cities, float optimal);

#ifdef __cplusplus
}
#endif

#endif
