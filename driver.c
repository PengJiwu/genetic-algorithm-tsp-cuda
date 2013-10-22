/* Driver for TSP solver using Genetic Algorithm.
 * Do not add any parallel construct specific code here.
 * An interface run() provided in tsp.h should be used to call
 * specific implementations.
 * This file only parses the arguments and loads the data */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

#include "tsp.h"

int main(int argc, char *argv[])
{
	/* Parse arguments - get the data file specifying TSP */
	int N = -1, maxgenerations = -1, maxpopulation = -1;
	float optimal = -1;
	/* check arguments and get data */
	if(argc < 2)
	{
		fprintf(stderr,"Usage : $ %s <tsp-city-data-file> [ p<population-size> ] "
				"[ g<generation-count> ] [ o<optimal-soln-known> ]\n", argv[0]);
		exit(1);
	}
	int i;
	for(i = 2 ; i < argc ; i++)
	{
		char * argument = argv[i];
		if(argument[0] == 'o')
			optimal = atof(++argument);
		else if(argument[0] == 'g')
			maxgenerations = atoi(++argument);
		else if(argument[0] == 'p')
			maxpopulation = atoi(++argument);
	}
	int num; float x,y;
	FILE *fp = fopen(argv[1], "r");
	if(!fp)
		exit(2);
	fscanf(fp, "%d\n", &N);	
	if(maxgenerations < 0)
		maxgenerations = MAX_GENERATIONS;
	if(maxpopulation < 0)
		maxpopulation = MAX_POPULATION;
	if(N < 0)
	{
		fprintf(stderr, "Please provide value of N\n");
		fprintf(stderr,"Usage : $ %s <tsp-city-data-file> n<city-count> [ p<population-size> ] "
				"[ g<generation-count> ] [ s<solution> ]\n", argv[0]);
		fclose(fp);
		exit(3);
	}
	city * cities = (city *) malloc(N * sizeof(struct _city));
	if(!cities)
	{
		perror("malloc");
		exit(4);
	}
	i = 0;
	while(fscanf(fp,"%d %f %f\n", &num, &x, &y) != EOF)
	{
		if(i > N)
		{
			fprintf(stderr, "There are more entries in data file than expected (%d).\n"
					"Truncating the remaining entries silently\n",N);
			break;
		}
		cities[i].x = x;
		cities[i].y = y;
		i++;
	}
	fclose(fp);

	int * result_tour = (int *) malloc(N * sizeof(int));
	int generations = run(cities, N, maxgenerations, maxpopulation, optimal, result_tour);
	fprintf(stderr, "Shortest tour path : %f\n", tour_length(result_tour, N, cities));
	fprintf(stderr, "Generations : %d\n", generations);
	plot_tour(result_tour, N, cities);
	return 0;
}

float tour_length(int * tour, int N, city * clist)
{
	float result = 0.0;
	float dx, dy;
	int i;
	for(i = 0 ; i < N ; i++)
	{
		dx = clist[ tour[(i + 1) % N] ].x - clist[ tour[i] ].x;
		dy = clist[ tour[(i + 1) % N] ].y - clist[ tour[i] ].y;
		result += sqrt((dx*dx) + (dy*dy));
	}
	return result;
}

void plot_tour(int * tour, int N, city * clist)
{
	int i;
	for(i = 0 ; i < N ; i++)
		fprintf(stdout, "%f %f\n", clist[tour[i]].x, clist[tour[i]].y);
}
