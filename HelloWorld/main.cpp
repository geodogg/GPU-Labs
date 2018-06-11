#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "functions.h"

using namespace std;

// argc - argument count is the number of parameters passed plus one more
//        parameter which is the name of the program that was executed. This is
//        held in the argv[0].
// argv - argument vector
int main(int argc, char * argv[]){
    printf("Running program: %s\n", argv[0]);
    printf("Hello! Welcome to the HelloWorld equivalent of CUDA.\n");

    int N = 1 << 20;  // approximately a million elements

    // ~~~~~~~~~~~~~~~ vector addition - HOST ~~~~~~~~~~~~~~~

    // generate vectors for addition
    float * c = new float[N]; // allocate memory of million floats on HOST
    if ( c == NULL ) exit (1);  // error check
    float * a = new float[N]; // allocate memory on HOST
    if ( a == NULL ) exit (1);  // error check
    float * b = new float[N]; // allocate memory on HOST
    if ( b == NULL ) exit (1);  // error check

    printf ("Number of bytes allocated per vector: %d bytes\n", N);

    printf("~~~~~~Vector addition on HOST~~~~~~\n");

    // initialize a and b on HOST
    for (int i = 0; i < N; i++){
      a[i] = 1.0f;
      b[i] = 2.0f;
    }

    clock_t tic = clock();  // start clocking

    add(N, c, a, b);  // vector addition on HOST

    printf("Addition of a[0] b[0] equals %f.\n", c[0]); // print single element output

    clock_t toc = clock() - tic;
    float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;

    printf("Vector addition on the HOST\nElapsed time: %f (sec)\n", elapsed_time);

    // free memory
    delete [] c;
    delete [] a;
    delete [] b;

    return 0;
}
