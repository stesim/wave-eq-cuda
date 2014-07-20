CUDA_INSTALL_PATH       ?= /opt/cuda
CUDA_SDK_INSTALL_PATH   ?= $(HOME)/NVIDIA_GPU_Computing_SDK

BIN						 = wave-eq-cuda
CXX						 = g++
SRC						 = $(wildcard *.cpp)
OBJ						 = $(SRC:%.cpp=%.o)
#OBJ						 = main.o cpusolver0.o cpusolver1.o cpusolver2.o cpusolver3.o
CXXFLAGS				 = -std=c++11 -march=native -O3
LIBS					 = -L$(CUDA_INSTALL_PATH)/lib64 -pthread -lcudart -lm
CXXDEPENDFILE			 = .cpp-depend

NVCC                    := $(CUDA_INSTALL_PATH)/bin/nvcc
NVSRC					 = $(wildcard *.cu)
NVOBJ					 = $(NVSRC:%.cu=%.o)
#NVOBJ					 = solver7.o solver9.o solver10.o solver11.o
INCLUDES                 = -I$(CUDA_INSTALL_PATH)/include
NVCCFLAGS				 = $(INCLUDES) -arch=sm_20 --ptxas-options=-v
NVDEPENDFILE			 = .cu-depend

all: $(BIN)

$(BIN): depend $(OBJ) $(NVOBJ)
	$(CXX) -o $@ $(OBJ) $(NVOBJ) $(LIBS)
	@echo " "
	@echo " "
	@echo "--------------------------------------------------------------------------------"
	@echo "> successfully built $(BIN)."
	@echo "--------------------------------------------------------------------------------"

$(OBJ): %.o : %.cpp
	@echo "--------------------------------------------------------------------------------"
	@echo "> compiling $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "> done." 
	@echo "--------------------------------------------------------------------------------"
	@echo " "
	@echo " "
	
$(NVOBJ): %.o : %.cu
	@echo "--------------------------------------------------------------------------------"
	@echo "> compiling $<"
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
	@echo "> done." 
	@echo "--------------------------------------------------------------------------------"
	@echo " "
	@echo " "

depend: 
	@echo "--------------------------------------------------------------------------------"
	@echo "> generating dependency files"
	@rm -f $(CXXDEPENDFILE) $(NVDEPENDFILE)
	@$(CXX) $(CXXFLAGS) -MM $(SRC) > $(CXXDEPENDFILE)
	@$(NVCC) $(NVCCFLAGS) -M $(NVSRC) > $(NVDEPENDFILE)
	@echo "> done." 
	@echo "--------------------------------------------------------------------------------"
	@echo " "
	@echo " "

-include $(CXXDEPENDFILE)

-include $(NVDEPENDFILE)

run: $(BIN)
	@echo "--------------------------------------------------------------------------------"
	@echo "> running $(BIN)"
	@echo "--------------------------------------------------------------------------------"
	./$(BIN)
	@echo "--------------------------------------------------------------------------------"
	
clean:
	@echo "--------------------------------------------------------------------------------"
	@echo "> cleaning up"
	@echo "--------------------------------------------------------------------------------"
	@rm -f $(OBJ) $(NVOBJ) $(CXXDEPENDFILE) $(NVDEPENDFILE) $(BIN)

.PHONY: depend run clean
#--opencc-options -OPT:Olimit=0 
################################################################################
#	End of File 
################################################################################

