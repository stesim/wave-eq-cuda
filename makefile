BIN = $(notdir $(shell pwd))
#SRC = main.cpp
SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.cpp.o)
CXXFLAGS = -std=c++11 -march=native -Wall -g #-O3
LIBS = -L/opt/cuda/lib64 -lcudart
CXX = g++
DEPENDFILE = .depend

NVSRC = $(wildcard *.cu)
NVOBJ = $(NVSRC:%.cu=%.cu.o)
NVCCFLAGS = -I/opt/cuda/include -lcudart -arch=sm_21 -fmad=false -O3
NVCC = nvcc

all: depend $(BIN)

$(BIN): $(OBJ) $(NVOBJ)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(OBJ) $(NVOBJ) $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

depend: 
	rm -f $(DEPENDFILE)
	$(CXX) $(CXXFLAGS) -MM $(SRC) > $(DEPENDFILE)

-include $(DEPENDFILE)

clean:
	rm -rf $(BIN) $(OBJ) $(NVOBJ)

force:
	make clean
	make

run: $(BIN)
	optirun ./$(BIN)

.PHONY: all depend clean