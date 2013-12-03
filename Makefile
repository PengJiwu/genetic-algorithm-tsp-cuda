CC = gcc 
NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -lm
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g
LDFLAGS =

all: tsp-cuda dummy

tsp-cuda: tsp_cuda.o driver.o warmup.o
	$(CC) tsp_cuda.o driver.o warmup.o $(CFLAGS) $(COPTFLAGS) -o tsp-cuda

dummy: dummy_tsp.o driver.o
	$(CC) dummy_tsp.o driver.o $(CFLAGS) $(COPTFLAGS) -o dummy

tsp_cuda.o: tsp_cuda.cu tsp_cuda.h tsp.h warmup.h
	$(NVCC) -c tsp_cuda.cu $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)

dummy_tsp.o: dummy_tsp.c tsp.h
	$(CC) -c dummy_tsp.c $(CFLAGS) $(COPTFLAGS)

driver.o: driver.c tsp.h
	$(CC) -c driver.c $(CFLAGS) $(COPTFLAGS)

warmup.o: warmup.cu warmup.h
	$(NVCC) -c warmup.cu $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)

clean:
	rm -f *.o dummy tsp-cuda prj-ga.* tour.dat tour.png
