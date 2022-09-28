CC=pgc++
CFLAGS=-acc -Minfo

all: laplace2d laplace2d_cpu

laplace2d:
	${CC} ${CFLAGS} laplace2d.cpp -o $@

laplace2d_cpu:
	${CC} ${CFLAGS} -ta=multicore laplace2d.cpp -o $@

clean:
	rm laplace2d
