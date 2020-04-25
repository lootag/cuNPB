#ifndef MATRIXMULTIPLY_H_
#define MATRIXMULTIPLY_H_
#include<vector>

bool MatrixMultiply(float * featureM, float * featureN, float * result, 
  int count_m, int count_n, int size, int gpu_id);
#endif