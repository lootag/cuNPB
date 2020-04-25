//#include "./MatrixMultiply.h"
#include <iostream>
#include "./preprocessing.h"
#include "MatrixMultiply.h"
#include <stdlib.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>

Eigen::MatrixXf Multiply(Eigen::MatrixXf Left, Eigen::MatrixXf Right)
{
    
    Eigen::MatrixXf toReturn(Left.rows(), Right.rows());
    if(Left.cols() != Right.cols())
    {
        std::cout << "The matrices you've given me are not conformable. Now I'll die :(" << std::endl;
        exit(1);
    }

    
    std::vector<std::vector<float>> A,B;
    
    for(int row = 0; row != Left.rows(); row++)
    {
        std::vector<float> row_to_push;
        for(int col = 0; col != Left.cols(); col ++)
        {
            row_to_push.push_back(Left(row,col));
        }
        A.push_back(row_to_push);
    }

    for(int row = 0; row != Right.rows(); row++)
    {
        std::vector<float> row_to_push;
        for(int col = 0; col != Right.cols(); col ++)
        {
            row_to_push.push_back(Right(row,col));
        }
        B.push_back(row_to_push);
    }
    
    int m = A.size();
    int n = B.size(); 
    int dim = A[1].size();
    int gpu_id = 0;

    float* feature_m = new float[ m*dim ];
    float* feature_n = new float[ n*dim ];
    auto tmp = feature_m;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dim; j++)
            *tmp++ = A[i][j];
    }

    tmp = feature_n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++)
            *tmp++ = B[i][j];
    }


    float* result = new float[m*n];
    MatrixMultiply(feature_m, feature_n, result, m, n, dim, gpu_id);
        
    for(int row = 0; row != toReturn.rows(); row++)
    {
        for(int col = 0; col != toReturn.cols(); col++)
        {
            toReturn(row, col) = result[col + row*toReturn.cols()];
        }
    }

    delete []feature_m;
    delete []feature_n;
    delete []result;
    

    return toReturn;


}