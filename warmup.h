#ifndef __WARMUP_H
#define __WARMUP_H

#define MAX_ITER 3000

#ifdef __cplusplus
extern "C" {
#endif

void warm_up(int grid, int block);
__global__ void warmup(float *data, int N);

#ifdef __cplusplus
}
#endif

#endif
