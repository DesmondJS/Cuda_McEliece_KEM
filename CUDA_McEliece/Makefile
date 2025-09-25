###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS= -I$(INC_DIR) $(CUDA_INC_DIR)
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS+= -I$(INC_DIR) $(CUDA_INC_DIR) -rdc=true
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcrypto -lssl

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o $(OBJ_DIR)/rng.o \
       $(OBJ_DIR)/shake_shim.o $(OBJ_DIR)/cuda_decrypt.o \
       $(OBJ_DIR)/fft_tables.o $(OBJ_DIR)/fft_tables_host.o $(OBJ_DIR)/vec.o \
       $(OBJ_DIR)/gf.o $(OBJ_DIR)/bm.o $(OBJ_DIR)/fft.o $(OBJ_DIR)/fft_tr.o

##########################################################

## Compile ##

# # Link c++ and CUDA compiled object files to target executable:
# $(EXE) : $(OBJS)
# 	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Final link: use nvcc instead of g++
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) $(CUDA_INC_DIR) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Compile rng.c to object file:
$(OBJ_DIR)/rng.o : $(INC_DIR)/rng.c $(INC_DIR)/rng.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile the OpenSSL SHAKE shim (in src/) as **C**
$(OBJ_DIR)/shake_shim.o : $(SRC_DIR)/shake_shim.c
	$(CC) -x c $(CC_FLAGS) -c $< -o $@

# Compile all those c files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -x c $(CC_FLAGS) -c $< -o $@


# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
