CXX = icc
CXX2 = gcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dp

all : $(TARGET1)
    
$(TARGET1) : dp.c kernel_dp.c dp.h
	$(CXX) $(CXXFLAGS2) -o $(TARGET1) dp.c kernel_dp.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
