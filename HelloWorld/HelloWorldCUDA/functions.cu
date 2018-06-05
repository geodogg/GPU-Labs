#include "functions.h"

void add(int n, float * p_sum, float * x, float * y){

  for (int i = 0; i < n; i++){
    p_sum[i] = x[i] + y[i];
  }
  return;
}.
