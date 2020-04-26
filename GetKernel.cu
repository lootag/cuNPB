#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <math.h>
#include <eigen3/Eigen/Dense>

using std::vector;

__global__ void getKernel(const float *a, const float *b, float *c, int cols) 
{
  // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
    if(row < cols && col < cols)
    {
        c[col + row * cols] = 0;
        for (int k = 0; k < cols; k++) 
        {
            c[col + row * cols] += (a[k + row * cols] - b[col + k * cols]) * (a[k + row * cols] - b[col + k * cols]);
        }
    }
  
  
}



int main() {
  // Matrix size of 4 x 3;
  int rows = 4;
  int cols = 3;
  Eigen::MatrixXf A(rows,cols);
  Eigen::MatrixXf C(rows,rows);
  int tmp = 1;
  for(int row = 0; row != A.rows(); row++)
  {
      for(int col = 0; col != A.cols(); col++)
      {
          A(row, col) = tmp;
          tmp += 1;
      }
  }

  

  // Size (in bytes) of matrix
  size_t bytes_a = rows * cols * sizeof(float);
  size_t bytes_b = cols * rows * sizeof(float);
  size_t bytes_c = rows * rows * sizeof(float);

  // Host vectors
  vector<float> h_a(rows * cols);
  vector<float> h_b(cols * rows);
  vector<float> h_c(rows * rows);

  for(int row = 0; row != A.rows(); row++)
  {
      for(int col = 0; col != A.cols(); col++)
      {
          h_a[col + row*cols] = A(row,col);
      }
  }

  for(int row = 0; row != A.transpose().rows(); row++)
  {
      for(int col = 0; col != A.transpose().cols(); col++)
      {
          h_b[col + row*A.transpose().cols()] = A.transpose()(row,col);
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
  int BLOCKS = 1;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  getKernel<<<blocks, threads>>>(d_a, d_b, d_c, rows);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  for(int row = 0; row != C.rows(); row++)
  {
      for(int col = 0; col != C.cols(); col++)
      {
          C(row,col) = h_c[col + row*C.cols()];
      }
  }

  for(int row = 0; row != C.rows(); row++)
  {
      for(int col = 0; col != C.cols(); col++)
      {
          std::cout << C(row,col) << std::endl;
      }
  }


  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}