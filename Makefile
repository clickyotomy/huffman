SHELL     = /bin/bash
PROG_NAME = huffman
CC        = clang
CFLAGS    = -Wall -Werror -Wextra -pedantic -std=c11
OBJS      = $(PROG_NAME).o tree.o map.o queue.o
FMT       = clang-format -style='{IndentWidth: 4,TabWidth: 4}' -i

default: $(PROG_NAME)

$(PROG_NAME): $(OBJS)
	@$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c
	@$(CC) $< $(CFLAGS) -c -o $@

format:
	@$(FMT) *.c *.h

test: default
	/usr/bin/time ./$(PROG_NAME) -e -i test/shakespeare.txt -o shakespeare.enc
	/usr/bin/time ./$(PROG_NAME) -d -i shakespeare.enc -o shakespeare.dec

	diff shakespeare.dec test/shakespeare.txt
	wc -c shakespeare.enc
	wc -c shakespeare.dec

clean:
	@/bin/rm -rf *~ *.o $(PROG_NAME) *.enc *.dec test.txt

.PHONY: default format test clean

