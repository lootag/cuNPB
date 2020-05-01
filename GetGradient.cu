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
    dsigma = Multiply(dsigma, dK_dsigma.transpose());
    dsigma = Multiply(dsigma, Kernel.inverse().transpose());
    dsigma = Multiply(dsigma, labels);
    dsigma = dsigma -0.5*Multiply(Kernel.inverse(), dK_dsigma.transpose()).trace();
    Eigen::MatrixXf dl = 0.5*Multiply(labels.transpose(), Kernel.inverse().transpose());
    dl = Multiply(dl, dK_dl.transpose());
    dl = Multiply(dl, Kernel.inverse().transpose());
    dl = Multiply(dl, labels);
    dl = dl -0.5*Multiply(Kernel.inverse(), dK_dl.transpose()).trace();
    gradient(0,0) = dsigma(0,0);
    gradient(0,1) = dl(0,0);
    return gradient;
}
