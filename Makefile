SHELL       = /bin/bash
PROG_NAME   = huffman
CC          = clang
CFLAGS      = -Wall -Werror -Wextra -pedantic -std=c11 -ggdb -O3
OBJS        = $(PROG_NAME).o tree.o map.o queue.o
FMT         = clang-format -style='{IndentWidth: 4,TabWidth: 4}' -i
VALGRIND    = valgrind --leak-check=full --show-leak-kinds=all
PERF_EVENTS = 'cache-references,cache-misses,cycles,instructions,branches,faults,migrations'
PERF_ARGS   = -B -e $(PERF_EVENTS)
PERF_STAT   = perf stat $(PERF_ARGS)
RAND_MIN    ?= 1024       # 1 kB.
RAND_MAX    ?= 4294967296 # 4 * 1024 * 1024 * 1024 bytes (~4GB).
RAND_AWK    = BEGIN{ srand(); print int(rand()*($(RAND_MAX)-$(RAND_MIN))+$(RAND_MIN)) }

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

test-perf: default
	$(PERF_STAT) -- ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	$(PERF_STAT) -- ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec


test-rand: default
	$(eval RAND_INT=$(shell awk '$(RAND_AWK)'))
	base64 /dev/urandom | head -c $(RAND_INT) >rand.txt
	/usr/bin/time ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	/usr/bin/time ./$(PROG_NAME) -d -i rand.enc -o rand.dec
	diff rand.txt rand.dec
	wc -c rand.enc
	wc -c rand.dec

mem-chk: default
	base64 /dev/urandom | head -c $(RAND_MIN) >rand.txt
	$(VALGRIND) -- ./$(PROG_NAME) -e -i rand.txt -o rand.enc
	$(VALGRIND) -- ./$(PROG_NAME) -d -i rand.enc -o rand.dec
	diff rand.txt rand.dec

clean:
	/bin/rm -rf *~ *.o $(PROG_NAME) *.enc *.dec rand.*


.PHONY: default format test test-perf test-rand mem-chk clean

