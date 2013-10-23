CC = gcc 
NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g
LDFLAGS =

all: tsp-cuda dummy

tsp-cuda: tsp_cuda.o driver.o
	$(CC) tsp_cuda.o driver.o $(CFLAGS) $(COPTFLAGS) -o tsp-cuda

dummy: dummy_tsp.o driver.o
	$(CC) dummy_tsp.o driver.o $(CFLAGS) $(COPTFLAGS) -o dummy

tsp_cuda.o: tsp_cuda.c tsp_cuda.h tsp.h
	$(CC) -c tsp_cuda.c $(CFLAGS) $(COPTFLAGS)

dummy_tsp.o: dummy_tsp.c tsp.h
	$(CC) -c dummy_tsp.c $(CFLAGS) $(COPTFLAGS)

driver.o: driver.c tsp.h
	$(CC) -c driver.c $(CFLAGS) $(COPTFLAGS)

clean:
	rm -f *.o dummy tsp-cuda
