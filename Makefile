CC=gcc #clang++ #g++
CXX=g++
NVCC=nvcc
NVFLAGS=-Xcompiler -rdynamic -G -g --gpu-architecture=compute_61 --gpu-code=sm_61 -expt-relaxed-constexpr --std=c++11 -Xcompiler "-O3 -march=x86-64 -mtune=native"
CXXFLAGS=-g -Wall -Wextra -std=c++11 -Weffc++
CFLAGS=-g -Wall -Wextra 
SOURCES=connected_components.cu
OBJ=connected_components.o
HEADER=
OBJDIR=.
EXE=concomp
INC=-I/usr/local/cuda/include/ -I./include
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart -lsparse_matrix_converter -lbebop_util
.SUFFIXES: .cpp .c .o .cu


all: $(OBJ) $(HEADER) $(SOURCES) $(EXE)

$(EXE): $(OBJ) $(HEADER) $(SOURCES)
	$(NVCC) $(NVFLAGS) $(DEFS) -Xcompiler="$(CXXFLAGS)" $(OBJ) -o $(EXE) $(LDFLAGS) -L.

.cpp.o: $(SOURCES) $(INC) $(HEADER)
	$(CXX) $(DEFS) $(INC) -c $(CXXFLAGS) $< -o $@

.c.o: $(SOURCES) $(INC) $(HEADER)
	$(CC) $(DEFS) $(INC) -c $(CFLAGS) $< -o $@

.cu.o: $(SOURCES) $(INC) $(HEADER)
	$(NVCC) $(NVFLAGS) $(DEFS) $(INC) -dc -Xcompiler="$(CXXFLAGS)" $< -o $@



clean:
	rm -f *.o $(EXE)
