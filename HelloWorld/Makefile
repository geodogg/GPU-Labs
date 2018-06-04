#	target: dependencies
#		action
hello: main.o functions.o
	clang++	main.o functions.o -o hello

main.o: main.cpp
	clang++ -c main.cpp

functions.o: functions.cpp functions.h
	clang++ -c functions.cpp

run: hello
		./hello

clean:
	rm hello *.o
