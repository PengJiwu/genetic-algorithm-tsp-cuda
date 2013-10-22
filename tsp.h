#ifndef __TSP_H
#define __TSP_H

#define MAX_GENERATIONS 200
#define MAX_POPULATION  1000

#define PERCENT_ERROR 0.5

typedef struct _city
{
	float x;
	float y;
}city;

/* A common interface to run different implementations of TSP solver genetic algorithm.
 * This method should be implemented in every implementation (e.g. cuda, p-threads, serial)
 * and then link that object with driver.o 
 * The function should return number of generations required */
int run(city * cities, int N, int maxgenerations, int maxpopulation, float optimal, int * result_tour);


float tour_length(int * tour, int N, city * clist);
void plot_tour(int * tour, int N, city * clist);

#endif
