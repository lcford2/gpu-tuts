CC=pgc++
CFLAGS=-acc -Minfo -std=c++11

all: laplace2d_gpu laplace2d_cpu

laplace2d_gpu:
	${CC} ${CFLAGS} laplace2d.cpp -o $@

laplace2d_cpu:
	${CC} ${CFLAGS} -ta=multicore laplace2d.cpp -o $@

clean:
	rm laplace2d_*

bsub-clean:
	rm *.out *.err
