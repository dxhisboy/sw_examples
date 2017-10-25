all: example
example: slave.o master.o libgptl.a
	mpicc -O2 -hybrid master.o slave.o libgptl.a -o example
slave.o: slave.c param.h
	sw5cc -slave -O2 -c slave.c -msimd
master.o: master.c param.h
	mpicc -host -O2 -c master.c
