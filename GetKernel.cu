#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <math.h>
#include "preprocessing.h"
#include <eigen3/Eigen/Dense>

using std::vector;

__global__ void getKernel(const float *a, const float *b, float *c, int rows, int cols, int rowsB) 
{
  // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
    if(row < rows && col < rowsB)
    {
        c[col + row * rowsB] = 0;
        for (int k = 0; k < cols; k++) 
        {
            c[col + row * rowsB] += (a[k + row * cols] - b[col + k * rowsB]) * (a[k + row * cols] - b[col + k * rowsB]);
        }
    }
  
  
}



Eigen::MatrixXf GetKernel(Eigen::MatrixXf A, Eigen::MatrixXf B) 
{
  // Matrix size of 4 x 3;
  int rows = A.rows();
  int cols = A.cols();
  int rowsB = B.rows();
  Eigen::MatrixXf C(rows,rowsB);
  

  

  // Size (in bytes) of matrix
  size_t bytes_a = rows * cols * sizeof(float);
  size_t bytes_b = cols * rowsB * sizeof(float);
  size_t bytes_c = rows * rowsB * sizeof(float);

  // Host vectors
  vector<float> h_a(rows * cols);
  vector<float> h_b(cols * rowsB);
  vector<float> h_c(rows * rowsB);

  for(int row = 0; row != A.rows(); row++)
  {
      for(int col = 0; col != A.cols(); col++)
      {
          h_a[col + row*cols] = A(row,col);
      }
  }

  for(int row = 0; row != B.transpose().rows(); row++)
  {
      for(int col = 0; col != B.transpose().cols(); col++)
      {
          h_b[col + row * B.transpose().cols()] = B.transpose()(row,col);
      }
  }

 


  


  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  float fTHREADS = THREADS;
  float frows = rows;
  float fBLOCKS = ceil(frows/fTHREADS);
  int BLOCKS = fBLOCKS;
  

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  getKernel<<<blocks, threads>>>(d_a, d_b, d_c, rows, cols, rowsB);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  for(int row = 0; row != C.rows(); row++)
  {
      for(int col = 0; col != C.cols(); col++)
      {
          C(row,col) = h_c[col + row*C.cols()];
      }
  }




  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return C;
}