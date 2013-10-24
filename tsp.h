#ifndef __TSP_H
#define __TSP_H


#define MAX_GENERATIONS 500
#define MAX_POPULATION  1000
#define MAX_ARR_LEN	200

#define PERCENT_ERROR 0.5

typedef struct _city
{
	float x;
	float y;
}city;

#ifdef __cplusplus
extern "C" {
#endif

/* A common interface to run different implementations of TSP solver genetic algorithm.
 * This method should be implemented in every implementation (e.g. cuda, p-threads, serial)
 * and then link that object with driver.o 
 * The function should return number of generations required */
int run(city * cities, int N, int maxgenerations, int maxpopulation, float optimal, unsigned int * result_tour);


float tour_length(unsigned int * tour, int N, city * clist);
void plot_tour(unsigned int * tour, int N, city * clist);
unsigned int * read_solution(char *filename, int N);

#ifdef __cplusplus
}
#endif

#endif
