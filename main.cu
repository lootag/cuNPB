#include <iostream>
#include "preprocessing.h"
#include "enumerators.h"
#include "optimizers.h"
#include <unistd.h>
#include <eigen3/Eigen/Dense>

int main()
{
    char *directory = "/content/src"; 
    int ret;
    ret = chdir(directory);
    //Eigen::MatrixXf Left = from_csv("train.csv", 4, 3);
    //Eigen::MatrixXf Right = from_csv("test.csv", 3, 2);
    
    /*
    Eigen::MatrixXf Result = Multiply(Left, Right.transpose());
    for(int row = 0; row != Result.rows(); row++)
    {
        for(int col = 0; col != Result.cols(); col++)
        {
            std::cout << Result(row, col) << std::endl;
        }
    }
    */
    Eigen::MatrixXf A(3,4);
    Eigen::MatrixXf B(3,1);
    float tmp = 1;
    for(int row = 0; row != A.rows(); row++)
    {
        for(int col = 0; col != A.cols(); col++)
        {
            A(row, col) = tmp;
            tmp += 1;
        }
    }
    tmp = 1;
    for(int row = 0; row != B.rows(); row++)
    {
        for(int col = 0; col != B.cols(); col++)
        {
            B(row, col) = tmp;
            tmp += 1;
        }
    }
    /*
    kernel_type type_standard = standard;
    Eigen::MatrixXf Result = GetKernel(A, A, 1, 1, type_standard);
    std::cout << "Standard Case" << std::endl;
    for(int row = 0; row != Result.rows(); row++)
    {
        for(int col = 0; col != Result.cols(); col++)
        {
            std::cout << Result(row, col) << std::endl;
        }
    }
    
    kernel_type type_dK_dsigma = dK_dsigma;

    Eigen::MatrixXf dK_dsigma = GetKernel(A, A, 1, 1, type_dK_dsigma);
    std::cout << "Derivative of K with respect to sigma" << std::endl;
    for(int row = 0; row != dK_dsigma.rows(); row++)
    {
        for(int col = 0; col != dK_dsigma.cols(); col++)
        {
            std::cout << dK_dsigma(row, col) << std::endl;
        }
    }

    kernel_type type_dK_dl2 = dK_dl;

    Eigen::MatrixXf dK_dl2 = GetKernel(A, A, 1, 1, type_dK_dl2);
    std::cout << "Derivative with respect to l2" << std::endl;
    for(int row = 0; row != dK_dl2.rows(); row++)
    {   
        for(int col = 0; col != dK_dl2.cols(); col++)
        {
            std::cout << dK_dl2(row, col) << std::endl;
        }
    }
    */
    /*
    Eigen::MatrixXf Gradient = GetGradient(A, B , 1, 1);    
    for(int row = 0; row != Gradient.rows(); row++)
    {
        for(int col = 0; col != Gradient.cols(); col++)
        {
            std::cout << Gradient(row, col) << std::endl;
        }
    }
    */
    float alpha = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    std::vector<float> parameters = Adam(alpha, beta1, beta2, A, B, 0.001, 10000);
    for(int index = 0; index != parameters.size(); index++)
    {
        std::cout << parameters[index] << std::endl; 
    }


    return 0;
}