#ifndef __TSP_CUDA_H
#define __TSP_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void get_random_tour(int **s, int index, int size);
int is_present(int * haystack, int needle, int size);
int is_acceptable(int * tour, int N, city * cities, float optimal);

#ifdef __cplusplus
}
#endif

#endif
