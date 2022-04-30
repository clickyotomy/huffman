export PATH := /usr/local/depot/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/depot/cuda-10.2/lib64/
SHELL       = /bin/bash
PROG_NAME   = huffman
CC          = $(shell which clang)
CPP         = $(shell which clang++)
CFLAGS      = -m64 -Wall -Werror -Wextra -ggdb -O3
OBJS        = $(PROG_NAME).o tree.o map.o queue.o parallel.o
LDFLAGS     = -L$(LD_LIBRARY_PATH) -lcudart
NVCC        = nvcc
NVCCFLAGS   = -m64 -O3 --gpu-architecture compute_61 -ccbin $(CPP)
FMT         = $(shell which clang-format) -style='{IndentWidth: 4, TabWidth: 4}' -i
VALGRIND    = valgrind --leak-check=full --show-leak-kinds=all
PERF_EVENTS = 'cache-references,cache-misses,cycles,instructions,branches,faults,migrations'
PERF_ARGS   = -B -e $(PERF_EVENTS)
PERF_STAT   = $(shell which perf) stat $(PERF_ARGS)
RAND_QMIN   = 0    # 0 bytes.
RAND_QMAX   = 128  # 128 bytes.
RAND_QAWK   = BEGIN{ srand(); print int(rand()*($(RAND_QMAX)-$(RAND_QMIN))+$(RAND_QMIN)) }
RAND_FMIN  ?= 1024       # 1 kB.
RAND_FMAX   ?= 536346624  # 512 MB.
# RAND_FMAX  ?= 4294967296 # 4 * 1024 * 1024 * 1024 bytes (~4GB).
RAND_FAWK   = BEGIN{ srand(); print int(rand()*($(RAND_FMAX)-$(RAND_FMIN))+$(RAND_FMIN)) }

default: $(PROG_NAME)


$(PROG_NAME): $(OBJS)
	$(CPP) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.cu
	PATH=$(PATH) $(NVCC) $(NVCCFLAGS) $(LDFLAGS) $< -c -o $@

%.o: %.c
	$(CC) $(CFLAGS) $< -c -o $@


format:
	$(FMT) *.c *.cu *.h


test: default
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec


test-mars: default
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -e -i test/mars.jpg -o mars.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -d -i mars.enc -o mars.dec


test-qrand: default
	$(eval RAND_INT=$(shell awk '$(RAND_QAWK)'))
	base64 /dev/urandom | head -c $(RAND_INT) >rand.txt

	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -d -i rand.enc -o rand.dec


test-frand: default
	$(eval RAND_INT=$(shell awk '$(RAND_FAWK)'))
	base64 /dev/urandom | head -c $(RAND_INT) >rand.txt

	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(PROG_NAME) -d -i rand.enc -o rand.dec


test-perf: default
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(PERF_STAT) -- ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(PERF_STAT) -- ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec


mem-chk: default
	base64 /dev/urandom | head -c $(RAND_FMIN) >rand.txt

	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(VALGRIND) -- ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(VALGRIND) -- ./$(PROG_NAME) -d -i rand.enc -o rand.dec


clean:
	/bin/rm -rf *~ *.o $(PROG_NAME) *.enc *.dec *.pdec rand.*


.PHONY: default format test test-qrand test-frand test-perf mem-chk clean

