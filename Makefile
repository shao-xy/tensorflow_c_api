USR_ROOT = /home/ceph/LSTM/libtensorflow
CXXFLAGS = -g -O3 -std=gnu++11 -I${USR_ROOT}/include -L${USR_ROOT}/lib -ltensorflow -ltensorflow_framework

.PHONY: all
all: multi-calc gen_inputdata test test_cpu

multi-calc: multi-calc.cc tf/TFContext.o tf/TestData.o
	#make -C tf/
	g++ $^ -o $@ ${CXXFLAGS}

gen_inputdata: gen_inputdata.cc tf/TestData.o
	g++ $^ -o $@ ${CXXFLAGS}

test: test.cpp
	g++ $^ -o $@ ${CXXFLAGS}

test_cpu: test_cpu.cc
	gcc -o $@ $^ -lpthread

.PHONY: clean
clean:
	rm -f test test_cpu multi-calc
	rm -f *.o
	make clean -C tf/
