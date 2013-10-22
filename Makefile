LDFLAGS= -lm
CCFLAGS= -g

dummy: dummy_tsp.o driver.o
	gcc dummy_tsp.o driver.o ${CCFLAGS} ${LDFLAGS} -o dummy

dummy_tsp.o: dummy_tsp.c tsp.h
	gcc -c dummy_tsp.c ${CCFLAGS} ${LDFLAGS}

driver.o: driver.c tsp.h
	gcc -c driver.c ${CCFLAGS} ${LDFLAGS}

clean:
	rm -f *.o dummy
