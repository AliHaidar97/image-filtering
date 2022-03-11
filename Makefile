SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
BFLAGS=-O3 -I$(HEADER_DIR)
NVFLAGS=$(BFLAGS) -Xcompiler -fopenmp
MPIFLAGS=$(BFLAGS) -fopenmp
LDFLAGS=-lm

SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	gif_load_store.c \
	cuda_sobelf.cu \
	main.c \
	openbsd-reallocarray.c \
	quantize.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/gif_load_store.o \
	$(OBJ_DIR)/cuda_sobelf.o \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/main.o : $(SRC_DIR)/main.c
	$(CC) -std=c11 $(MPIFLAGS) -c -o $@ $^

$(OBJ_DIR)/cuda_sobelf.o : $(SRC_DIR)/cuda_sobelf.cu
	nvcc $(NVFLAGS) -c -o $@ $^

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(MPIFLAGS) -c -o $@ $^

sobelf:$(OBJ)
	$(CC) $(MPIFLAGS) -o $@ $^ $(LDFLAGS) -lcudart -L/usr/local/cuda/lib64

clean:
	rm -f sobelf $(OBJ)
