CC=pgc++
CFLAGS=-acc -Minfo

all: saxpy saxpy_cpu

saxpy:
	${CC} ${CFLAGS} saxpy.cpp -o $@

saxpy_cpu:
	${CC} ${CFLAGS} -ta=multicore saxpy.cpp -o $@

clean:
	rm saxpy saxpy_cpu
