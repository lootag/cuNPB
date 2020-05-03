#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include "preprocessing.h"

void to_csv(Eigen::MatrixXf Matrix, std::string file_name)
{
    std::ofstream csv_file;
    csv_file.open(file_name);
    for(int row = 0; row != Matrix.rows(); row++ )
    {
        for(int col = 0; col != Matrix.cols(); col++)
        {
            if(col < (Matrix.cols() -1))
            {
                csv_file << std::to_string(Matrix(row,col)) + ",";
            }
            else
            {
                csv_file << std::to_string(Matrix(row,col)) + "/n";
            }
            
        }
    }
}