all: test.cpp allocators.hpp
	nvcc -O3 -std=c++11 -ccbin `which g++` -x cu -arch sm_70 test.cpp -o test
#	g++ -O3 -std=c++11 -o test test.cpp
