LDFLAGS= -lm
CCFLAGS= -g

all: tsp-cuda dummy

tsp-cuda: tsp_cuda.o driver.o
	gcc tsp_cuda.o driver.o ${CCFLAGS} ${LDFLAGS} -o tsp-cuda

dummy: dummy_tsp.o driver.o
	gcc dummy_tsp.o driver.o ${CCFLAGS} ${LDFLAGS} -o dummy

tsp_cuda.o: tsp_cuda.c tsp_cuda.h tsp.h
	gcc -c tsp_cuda.c ${CCFLAGS} ${LDFLAGS}

dummy_tsp.o: dummy_tsp.c tsp.h
	gcc -c dummy_tsp.c ${CCFLAGS} ${LDFLAGS}

driver.o: driver.c tsp.h
	gcc -c driver.c ${CCFLAGS} ${LDFLAGS}

clean:
	rm -f *.o dummy tsp-cuda
