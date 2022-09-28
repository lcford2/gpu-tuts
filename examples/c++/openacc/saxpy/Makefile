CC=pgc++
CFLAGS=-acc -Minfo -std=c++11

all: saxpy_gpu saxpy_cpu

saxpy_gpu:
	${CC} ${CFLAGS} saxpy.cpp -o $@

saxpy_cpu:
	${CC} ${CFLAGS} -ta=multicore saxpy.cpp -o $@

clean:
	rm saxpy_*
