#include <iostream>
#include <eigen3/Eigen/Dense>
#include "preprocessing.h"
#include "enumerators.h"
#include <vector>

Eigen::MatrixXf GetGradient(Eigen::MatrixXf Train, Eigen::MatrixXf labels, float sigma, float l)
{
    Eigen::MatrixXf gradient(2,1);
    kernel_type type_standard = standard;
    kernel_type type_dK_dsigma = dK_dsigma;
    kernel_type type_dK_dl = dK_dl;
    Eigen::MatrixXf Kernel = GetKernel(Train, Train, sigma, l, type_standard);   
    Eigen::MatrixXf dK_dsigma = GetKernel(Train, Train, sigma, l, type_dK_dsigma);   
    Eigen::MatrixXf dK_dl = GetKernel(Train, Train, sigma, l, type_dK_dl);   
    Eigen::MatrixXf dsigma = 0.5*Multiply(labels.transpose(), Kernel.inverse().transpose());
    std::cout << "Kernel" << std::endl;
    for (int row = 0; row != Kernel.inverse().rows())
    {
        for(int col = 0; col != Kernel.inverse().cols())
        {
            std::cout << Kernel.inverse()(row, col) << std::endl;
        }
    }
    std::cout << "labels.transpose()" << std::endl;
    for(int row = 0; row != labels.transpose().rows(); row++)
    {
        for(int col = 0; col != labels.transpose().cols(); col++)
        {
            std::cout << labels.transpose()(row,col) << std::endl;
        }
    }
    /*
    dsigma = Multiply(dsigma, dK_dsigma.transpose());
    dsigma = Multiply(dsigma, Kernel.inverse().transpose());
    dsigma = Multiply(dsigma, labels);
    Eigen::MatrixXf trace_dsigma(1,1);
    trace_dsigma(0,0) = 0.5*Multiply(Kernel.inverse(), dK_dsigma.transpose()).trace();
    dsigma = dsigma - trace_dsigma;
    Eigen::MatrixXf dl = 0.5*Multiply(labels.transpose(), Kernel.inverse().transpose());
    dl = Multiply(dl, dK_dl.transpose());
    dl = Multiply(dl, Kernel.inverse().transpose());
    dl = Multiply(dl, labels);
    Eigen::MatrixXf trace_dl(1,1);
    trace_dl(0,0) = 0.5*Multiply(Kernel.inverse(), dK_dl.transpose()).trace();
    dl = dl - trace_dl;
    gradient(0,0) = dsigma(0,0);
    gradient(0,1) = dl(0,0);
    */
    std::cout << "dsigma" << std::enld; 
    return dsigma;
}
