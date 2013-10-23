#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tsp.h"
#include "tsp_cuda.h"

int run(city * cities, int N, int maxgenerations, int maxpopulation, float optimal, int * result_tour)
{
	int **s1, **s2, *p1, *p2;
	s1 = (int **) malloc(maxpopulation * sizeof(int *));
	int i,j;
	/* initialize population with random tours */
	for(i = 0 ; i < maxpopulation ; i++)
	{
		s1[i] = (int *) malloc(N * sizeof(int));
		get_random_tour(s1, i, N);
		if(optimal >= 0)	/* an optimal solution has been provided. */
		{
			if(is_acceptable(s1[i], N, cities, optimal))
			{
				for(j = 0 ; j < N ; j++)
					result_tour[j] = s1[i][j];
				return 0;
			}
		}
	}

	/* Temp dummy code here */
	for(j = 0 ; j < N ; j++)
		result_tour[j] = s1[0][j];

	return -1;
}

void get_random_tour(int **s, int index, int size)
{
	int * tour = s[index];
	/* set seed depending on current time */
	srandom(time(NULL));
	
	/* generate (pseudo-)random tour */
	int i,j;
	unsigned int r;
	for(i = 0 ; i < size ; i++)
	{
		r = random() % size;
		while(is_present(tour, r, i))			/* TODO : pretty crude way of generating unique numbers in 0..N. */
			r = random() % size;
		tour[i] = r;
	}
}

int is_present(int * haystack, int needle, int size)		// This is simple linear search. TODO : binary search if reqd.
{
	int i;
	for(i = 0 ; i < size ; i++)
	{
		if(haystack[i] == needle)
			return 1;
	}
	return 0;
}

int is_acceptable(int * tour, int N, city * cities, float optimal)
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
