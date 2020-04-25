#include <iostream>
#include "preprocessing.h"
#include <unistd.h>
#include <eigen3/Eigen/Dense>

int main()
{
    char *directory = "/content/src"; 
    int ret;
    ret = chdir(directory);
    Eigen::MatrixXf Left = from_csv("train.csv", 4, 3);
    Eigen::MatrixXf Right = from_csv("test.csv", 3, 2);
    
    Eigen::MatrixXf Result = Multiply(Left, Right.transpose());
    for(int row = 0; row != Result.rows(); row++)
    {
        for(int col = 0; col != Result.cols(); col++)
        {
            std::cout << Result(row, col) << std::endl;
        }
    }


    return 0;
}