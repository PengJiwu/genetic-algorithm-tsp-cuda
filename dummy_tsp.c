#include <stdio.h>
#include <stdlib.h>

#include "tsp.h"

int run(city * cities, int N, int maxgenerations, int maxpopulation, float optimal, int * result_tour)
{
	/* This file doesn't have a real implementation. This is just a stub -
	 * 1. to help derive other implementations
	 * 2. to allow the driver code to compile and run independently */
	int i;
	for(i = 0 ; i < N ; i++)
		result_tour[i] = random() % N;
	return 0;
}
