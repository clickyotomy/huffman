SHELL       = /bin/bash
PROG_NAME   = huffman
CC          = g++
CFLAGS      = -m64 -Wall -Wextra -pedantic -ggdb -O3
LDFLAGS     =-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
OBJS        = $(PROG_NAME).o tree.o map.o queue.o decode_parallel.o
NVCC        =nvcc
NVCCFLAGS   =-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
CU_DEPS     =
CU_FILES    = decode_parallel.cu
FMT         = clang-format -style='{IndentWidth: 4,TabWidth: 4}' -i
VALGRIND    = valgrind --leak-check=full --show-leak-kinds=all
PERF_EVENTS = 'cache-references,cache-misses,cycles,instructions,branches,faults,migrations'
PERF_ARGS   = -B -e $(PERF_EVENTS)
PERF_STAT   = perf stat $(PERF_ARGS)
RAND_QMIN   = 1	  # 1 byte.
RAND_QMAX   = 128 # 128 bytes.
RAND_QAWK   = BEGIN{ srand(); print int(rand()*($(RAND_QMAX)-$(RAND_QMIN))+$(RAND_QMIN)) }
RAND_FMIN   ?= 1024       # 1 kB.
RAND_FMAX   ?= 4294967296 # 4 * 1024 * 1024 * 1024 bytes (~4GB).
RAND_FAWK   = BEGIN{ srand(); print int(rand()*($(RAND_FMAX)-$(RAND_FMIN))+$(RAND_FMIN)) }

default: $(PROG_NAME)


$(PROG_NAME): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)


%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@


format:
	$(FMT) *.c *.h


test: default
	/usr/bin/time ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	/usr/bin/time ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec

	diff shakespeare.dec test/shakespeare.txt

	wc -c shakespeare.enc
	wc -c shakespeare.dec


test-qrand: default
	$(eval RAND_INT=$(shell awk '$(RAND_QAWK)'))
	base64 /dev/urandom | head -c $(RAND_INT) >rand.txt

	./$(PROG_NAME) -e -i rand.txt -o rand.enc
	./$(PROG_NAME) -d -i rand.enc -o rand.dec

	diff rand.dec rand.txt


test-frand: default
	$(eval RAND_INT=$(shell awk '$(RAND_FAWK)'))
	base64 /dev/urandom | head -c $(RAND_INT) >rand.txt

	/usr/bin/time ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	/usr/bin/time ./$(PROG_NAME) -d -i rand.enc -o rand.dec

	diff rand.dec rand.txt

	wc -c rand.enc
	wc -c rand.dec


test-perf: default
	$(PERF_STAT) -- ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	$(PERF_STAT) -- ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec


mem-chk: default
	base64 /dev/urandom | head -c $(RAND_FMIN) >rand.txt

	$(VALGRIND) -- ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	$(VALGRIND) -- ./$(PROG_NAME) -d -i rand.enc -o rand.dec

	diff rand.dec rand.txt


clean:
	/bin/rm -rf *~ *.o $(PROG_NAME) *.enc *.dec rand.*


.PHONY: default format test test-qrand test-frand test-perf mem-chk clean

