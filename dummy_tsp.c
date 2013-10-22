#include <stdio.h>
#include <stdlib.h>

#include "tsp.h"

void run(city * cities, int N, int maxgenerations, float optimal, int * result_tour)
{
	/* This file doesn't have a real implementation. This is just a stub to
	 * allow the driver.c to compile and run */
	int i;
	for(i = 0 ; i < N ; i++)
		result_tour[i] = random() % N;
}
